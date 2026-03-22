"""Paper parsing: combines cache (LaTeX source), references, and metadata.

Takes an arxiv_id and assembles a ParsedPaper from:
  - Cache: extracted plain text + .tex/.bib/.bbl files
  - References module: structured reference extraction from LaTeX source
  - Metadata index: title, abstract, authors (from Kaggle dataset)

References without arxiv IDs get resolved via title matching against the
metadata index.
"""

import logging
from dataclasses import dataclass, field

from .cache import PaperCache
from .metadata import ArxivMetadata
from .references import Reference, extract_references

logger = logging.getLogger(__name__)


@dataclass
class ParsedPaper:
    """Fully parsed paper with text, metadata, and resolved references."""

    arxiv_id: str
    full_text: str
    title: str | None = None
    abstract: str | None = None
    authors: str | None = None
    categories: str | None = None
    references: list[Reference] = field(default_factory=list)


def parse_paper(
    arxiv_id: str,
    cache: PaperCache,
    metadata: ArxivMetadata | None = None,
) -> ParsedPaper | None:
    """Parse a paper from cached LaTeX source + metadata.

    Args:
        arxiv_id: The paper's arxiv ID.
        cache: PaperCache with downloaded LaTeX source.
        metadata: Optional ArxivMetadata for title/abstract and ref resolution.

    Returns:
        ParsedPaper, or None if source is not available.
    """
    text = cache.get_text(arxiv_id)
    if text is None:
        return None

    # Get metadata from Kaggle index
    title = None
    abstract = None
    authors = None
    categories = None
    if metadata is not None:
        meta = metadata.get(arxiv_id)
        if meta:
            title = meta["title"]
            abstract = meta["abstract"]
            authors = meta["authors"]
            categories = meta["categories"]

    # Extract references from .tex + .bib + .bbl files
    tex_files = cache.get_tex_files(arxiv_id)
    bib_files = cache.get_bib_files(arxiv_id)
    bbl_files = cache.get_bbl_files(arxiv_id)
    references = extract_references(tex_files, bib_files, bbl_files)

    # Resolve references without arxiv IDs via title matching
    if metadata is not None:
        resolved = 0
        for ref in references:
            if ref.arxiv_id is None and ref.title:
                matched_id = metadata.resolve_title(ref.title)
                if matched_id:
                    ref.arxiv_id = matched_id
                    resolved += 1
        if resolved > 0:
            logger.info("%s: resolved %d refs via title matching", arxiv_id, resolved)

    logger.info(
        "%s: %s, %d refs (%d with arxiv IDs)",
        arxiv_id,
        title[:60] if title else "(no title)",
        len(references),
        sum(1 for r in references if r.arxiv_id),
    )

    return ParsedPaper(
        arxiv_id=arxiv_id,
        full_text=text,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        references=references,
    )
