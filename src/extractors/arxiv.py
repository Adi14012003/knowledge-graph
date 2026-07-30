import hashlib
import re
import arxiv
from src.models import Document, SourceType

def _extract_arxiv_id(url_or_id: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5})", url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arXiv ID from: {url_or_id}")

def fetch_arxiv(url_or_id: str) -> Document:
    paper_id = _extract_arxiv_id(url_or_id)
    
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    results = list(client.results(search))
    
    if not results:
        raise ValueError(f"No paper found for ID: {paper_id}")
    
    paper = results[0]
    authors = ", ".join(str(a) for a in paper.authors)
    raw_text = f"""Title: {paper.title}
Authors: {authors}
Published: {paper.published.strftime('%Y-%m-%d')}
Abstract:
{paper.summary}
Categories: {', '.join(paper.categories)}"""
    
    canonical_url = f"https://arxiv.org/abs/{paper_id}"
    content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
    doc_id = hashlib.sha256(canonical_url.encode()).hexdigest()[:16]
    
    return Document(
        id=doc_id,
        source_url=canonical_url,
        source_type=SourceType.ARXIV,
        title=paper.title,
        raw_text=raw_text,
        metadata={"authors": authors, "arxiv_id": paper_id},
        content_hash=content_hash,
    )
