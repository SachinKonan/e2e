"""PDF parsing and reference extraction.

Uses pymupdf (fitz) for text extraction and regex-based reference parsing.
Falls back to GROBID-style heuristics for structured reference extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf

logger = logging.getLogger(__name__)

# Patterns to match arxiv IDs in reference text
ARXIV_ID_PATTERNS = [
    re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE),
    re.compile(r"arXiv[:\s]*([a-z-]+/\d{7}(?:v\d+)?)", re.IGNORECASE),
    re.compile(r"(?:https?://)?arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE),
    re.compile(r"(?:https?://)?arxiv\.org/(?:abs|pdf)/([a-z-]+/\d{7}(?:v\d+)?)", re.IGNORECASE),
]

# Pattern to find reference section
REF_SECTION_PATTERN = re.compile(
    r"\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
    re.IGNORECASE,
)

# Pattern to match individual references (e.g., [1], [2], etc.)
REF_ENTRY_PATTERN = re.compile(r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|\Z)", re.DOTALL)


@dataclass
class Reference:
    """A parsed reference from a paper."""

    raw_text: str
    arxiv_id: str | None = None
    title: str | None = None
    authors: str | None = None
    year: str | None = None
    ref_number: int | None = None


@dataclass
class ParsedPaper:
    """Result of parsing a PDF."""

    arxiv_id: str
    full_text: str
    title: str | None = None
    abstract: str | None = None
    references: list[Reference] = field(default_factory=list)
    reference_section_text: str | None = None


def extract_text(pdf_path: Path) -> str:
    """Extract full text from a PDF using pymupdf."""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def extract_title(text: str) -> str | None:
    """Heuristic: first non-empty line that looks like a title."""
    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        # Skip very short lines or lines that look like headers/page numbers
        if len(line) > 10 and not line.isdigit() and not line.startswith("arXiv:"):
            return line
    return None


def extract_abstract(text: str) -> str | None:
    """Extract abstract section."""
    match = re.search(
        r"(?:Abstract|ABSTRACT)[.\s]*\n(.*?)(?:\n\s*(?:1[\s.]|Introduction|INTRODUCTION|I\.\s))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def extract_arxiv_ids_from_text(text: str) -> list[str]:
    """Find all arxiv IDs mentioned in text."""
    ids = []
    for pattern in ARXIV_ID_PATTERNS:
        for match in pattern.finditer(text):
            aid = match.group(1)
            # Strip version suffix for dedup
            aid = re.sub(r"v\d+$", "", aid)
            if aid not in ids:
                ids.append(aid)
    return ids


def extract_references(text: str) -> tuple[list[Reference], str | None]:
    """Extract references section and parse individual entries.

    Returns:
        Tuple of (list of References, raw reference section text).
    """
    # Find reference section
    ref_match = REF_SECTION_PATTERN.search(text)
    if not ref_match:
        logger.warning("No reference section found")
        return [], None

    ref_text = text[ref_match.start():]
    references = []

    # Try numbered references first [1], [2], etc.
    entries = REF_ENTRY_PATTERN.findall(ref_text)
    if entries:
        for num_str, entry_text in entries:
            entry_text = entry_text.strip()
            entry_text = re.sub(r"\s+", " ", entry_text)  # normalize whitespace

            ref = Reference(
                raw_text=entry_text,
                ref_number=int(num_str),
            )

            # Try to extract arxiv ID from the reference text
            arxiv_ids = extract_arxiv_ids_from_text(entry_text)
            if arxiv_ids:
                ref.arxiv_id = arxiv_ids[0]

            # Try to extract year
            year_match = re.search(r"\b(19|20)\d{2}\b", entry_text)
            if year_match:
                ref.year = year_match.group(0)

            references.append(ref)
    else:
        # Fallback: just extract any arxiv IDs from the reference section
        arxiv_ids = extract_arxiv_ids_from_text(ref_text)
        for aid in arxiv_ids:
            references.append(Reference(raw_text=aid, arxiv_id=aid))

    return references, ref_text


def parse_pdf(pdf_path: Path, arxiv_id: str) -> ParsedPaper:
    """Parse a PDF and extract structured information.

    Args:
        pdf_path: Path to the PDF file.
        arxiv_id: The arxiv ID of the paper.

    Returns:
        ParsedPaper with text, title, abstract, and references.
    """
    text = extract_text(pdf_path)
    title = extract_title(text)
    abstract = extract_abstract(text)
    references, ref_section = extract_references(text)

    logger.info(
        "Parsed %s: title=%s, %d refs (%d with arxiv IDs)",
        arxiv_id,
        title[:50] if title else None,
        len(references),
        sum(1 for r in references if r.arxiv_id),
    )

    return ParsedPaper(
        arxiv_id=arxiv_id,
        full_text=text,
        title=title,
        abstract=abstract,
        references=references,
        reference_section_text=ref_section,
    )
