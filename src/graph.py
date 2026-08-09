import os
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2000)

def clean(text: str) -> str:
    if not text:
        return ""
    return text.encode('utf-8', errors='ignore').decode('utf-8').replace('\u2028', ' ').replace('\u2029', ' ').replace('\u0000', '')

class AgentState(TypedDict):
    query: str
    query_type: str
    retrieved_chunks: list[str]
    retrieved_entities: list[dict]
    draft_answer: str
    final_answer: str
    sources: list[str]
    gaps_filled: list[str]
    critic_passed: bool
    research_attempts: int
    error: str

def router_node(state: AgentState) -> AgentState:
    prompt = f"""Classify this query into exactly one of these:
- simple_lookup
- graph_query
- research_needed

Query: {state['query']}
Reply with just the category name."""
    response = llm.invoke(prompt)
    query_type = response.content.strip().lower()
    if query_type not in ["simple_lookup", "graph_query", "research_needed"]:
        query_type = "simple_lookup"
    print(f"[Router] Query type: {query_type}")
    return {**state, "query_type": query_type}

def retriever_node(state: AgentState) -> AgentState:
    print(f"[Retriever] Searching for: {state['query']}")
    from src.storage.neo4j_store import search_entities, get_related_entities
    direct_matches = search_entities(state["query"], limit=8)
    graph_context = []
    for match in direct_matches[:3]:
        related = get_related_entities(match["name"])
        if related:
            names = ", ".join(r["name"] for r in related[:5])
            graph_context.append(f"{match['name']} connects to: {names}")
    chunks = []
    for e in direct_matches:
        chunks.append(clean(f"[{e['type'].upper()}] {e['name']}: {e['description']} (subtopic: {e['subtopic']})"))
    if graph_context:
        chunks.append("Graph connections: " + " | ".join(graph_context))
    if not chunks:
        chunks = ["No matching entities found in knowledge base."]
    print(f"[Retriever] Found {len(direct_matches)} entities")
    return {**state, "retrieved_chunks": chunks, "retrieved_entities": direct_matches}

def research_node(state: AgentState) -> AgentState:
    if state["research_attempts"] >= 3:
        return state
    print(f"[Research] Checking for gaps...")
    prompt = f"""A user asked: "{state['query']}"
Current knowledge base has {len(state['retrieved_chunks'])} results.
Is there a specific arXiv paper ID that would significantly improve the answer?
Reply with just the paper ID like 2305.10403, or reply "sufficient"."""
    response = llm.invoke(prompt)
    suggestion = response.content.strip()
    if suggestion.lower() == "sufficient" or not suggestion:
        return state
    try:
        from src.extractors.arxiv import fetch_arxiv
        from src.extraction import extract_entities
        from src.storage.neo4j_store import save_entities as neo4j_save
        from src.storage.qdrant_store import save_entities as qdrant_save
        doc = fetch_arxiv(suggestion)
        result = extract_entities(doc)
        if result.is_relevant and result.entities:
            neo4j_save(result)
            qdrant_save(result)
        gaps_filled = state.get("gaps_filled", []) + [doc.source_url]
        new_chunk = clean(f"[Newly fetched: {doc.title}]\n{doc.raw_text[:600]}")
        print(f"[Research] Fetched: {doc.title}")
        return {
            **state,
            "retrieved_chunks": state["retrieved_chunks"] + [new_chunk],
            "gaps_filled": gaps_filled,
            "research_attempts": state["research_attempts"] + 1,
        }
    except Exception as e:
        print(f"[Research] Failed: {e}")
        return {**state, "research_attempts": state["research_attempts"] + 1}

def synthesis_node(state: AgentState) -> AgentState:
    print(f"[Synthesis] Writing answer...")
    sources_text = "\n".join(state["retrieved_chunks"][:8])
    prompt = f"""Answer this question using the retrieved knowledge below.
Add a citation [Source: entity name] for each key claim.

Question: {state['query']}

Retrieved knowledge:
{sources_text}"""
    response = llm.invoke(prompt)
    return {**state, "draft_answer": clean(response.content)}

def critic_node(state: AgentState) -> AgentState:
    print(f"[Critic] Checking answer...")
    if not state.get("draft_answer"):
        return {**state, "critic_passed": False}
    prompt = f"""Check this answer against its sources.
Answer: {state['draft_answer']}
Sources: {chr(10).join(state['retrieved_chunks'][:5])}
Reply APPROVED if all claims are supported, or REJECTED: [reason]"""
    response = llm.invoke(prompt)
    passed = response.content.strip().upper().startswith("APPROVED")
    print(f"[Critic] {'APPROVED' if passed else 'REJECTED'}")
    return {**state, "critic_passed": passed, "final_answer": state["draft_answer"]}

def after_router(state: AgentState) -> Literal["retriever"]:
    return "retriever"

def after_retriever(state: AgentState) -> Literal["research", "synthesis"]:
    if state["query_type"] == "research_needed" and state["research_attempts"] < 2:
        return "research"
    return "synthesis"

def after_research(state: AgentState) -> Literal["synthesis"]:
    return "synthesis"

def after_synthesis(state: AgentState) -> Literal["critic"]:
    return "critic"

def after_critic(state: AgentState) -> Literal["__end__"]:
    return "__end__"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router)
    graph.add_conditional_edges("retriever", after_retriever)
    graph.add_conditional_edges("research", after_research)
    graph.add_conditional_edges("synthesis", after_synthesis)
    graph.add_conditional_edges("critic", after_critic)
    return graph.compile()

kg_graph = build_graph()

def run_query(question: str) -> dict:
    initial_state: AgentState = {
        "query": question,
        "query_type": "",
        "retrieved_chunks": [],
        "retrieved_entities": [],
        "draft_answer": "",
        "final_answer": "",
        "sources": [],
        "gaps_filled": [],
        "critic_passed": False,
        "research_attempts": 0,
        "error": "",
    }
    result = kg_graph.invoke(initial_state)
    return {
        "answer": result.get("final_answer") or result.get("draft_answer", ""),
        "sources": result.get("sources", []),
        "gaps_filled": result.get("gaps_filled", []),
        "critic_passed": result.get("critic_passed", False),
        "entities_used": len(result.get("retrieved_entities", [])),
    }
