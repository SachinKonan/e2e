"""BibTeX and .bbl reference parsing.

Extracts structured references from LaTeX source:
  1. .bib files  → BibTeX entries with title, author, year, arxiv ID (eprint field)
  2. .bbl files  → Compiled bibliography (\\bibitem entries)
  3. .tex files  → \\cite{} commands to find which refs are actually used

We combine all three to build a complete reference list, then resolve
titles → arxiv IDs using the Kaggle metadata index.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# --- BibTeX parsing ---

# Match @type{key, ... }
_BIBTEX_ENTRY = re.compile(
    r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\s*\}",
    re.DOTALL,
)

# Match field = {value} or field = "value" or field = number
_BIBTEX_FIELD = re.compile(
    r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))",
    re.DOTALL,
)

# --- .bbl parsing ---

_BIBITEM = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)",
    re.DOTALL,
)

# --- .tex citation parsing ---

_CITE_CMD = re.compile(r"\\(?:cite|citep|citet|citealt|citealp|citeauthor|citeyear)\s*(?:\[[^\]]*\])?\{([^}]+)\}")

# --- arxiv ID extraction ---

_ARXIV_EPRINT = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
_ARXIV_OLD = re.compile(r"([a-z-]+/\d{7}(?:v\d+)?)")


@dataclass
class Reference:
    """A parsed reference from a paper."""

    cite_key: str
    raw_text: str = ""
    title: str | None = None
    authors: str | None = None
    year: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    source: str = ""  # "bib", "bbl", or "inline"


def _clean_bibtex_value(val: str) -> str:
    """Clean up a BibTeX field value."""
    val = val.strip()
    val = re.sub(r"\s+", " ", val)
    val = val.replace("{", "").replace("}", "")
    return val


def parse_bib(bib_text: str) -> list[Reference]:
    """Parse a .bib file into Reference objects."""
    refs = []
    for match in _BIBTEX_ENTRY.finditer(bib_text):
        entry_type = match.group(1).lower()
        cite_key = match.group(2)
        body = match.group(3)

        if entry_type in ("string", "preamble", "comment"):
            continue

        fields = {}
        for fm in _BIBTEX_FIELD.finditer(body):
            field_name = fm.group(1).lower()
            value = fm.group(2) or fm.group(3) or fm.group(4) or ""
            fields[field_name] = _clean_bibtex_value(value)

        ref = Reference(
            cite_key=cite_key,
            raw_text=body.strip()[:500],
            title=fields.get("title"),
            authors=fields.get("author"),
            year=fields.get("year"),
            doi=fields.get("doi"),
            source="bib",
        )

        # Check for arxiv ID in eprint, url, or note fields
        for field_name in ("eprint", "url", "note", "archiveprefix"):
            val = fields.get(field_name, "")
            m = _ARXIV_EPRINT.search(val)
            if m:
                ref.arxiv_id = re.sub(r"v\d+$", "", m.group(1))
                break
            m = _ARXIV_OLD.search(val)
            if m:
                ref.arxiv_id = re.sub(r"v\d+$", "", m.group(1))
                break

        refs.append(ref)

    return refs


def parse_bbl(bbl_text: str) -> list[Reference]:
    """Parse a .bbl file (compiled bibliography) into Reference objects."""
    refs = []
    for match in _BIBITEM.finditer(bbl_text):
        cite_key = match.group(1)
        body = match.group(2).strip()

        # Clean up LaTeX from the body
        clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", body)
        clean = re.sub(r"[{}\\]", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        ref = Reference(
            cite_key=cite_key,
            raw_text=clean[:500],
            source="bbl",
        )

        # Try to extract year
        year_match = re.search(r"\b(19|20)\d{2}\b", clean)
        if year_match:
            ref.year = year_match.group(0)

        # Try to extract arxiv ID
        for pattern in (_ARXIV_EPRINT, _ARXIV_OLD):
            m = pattern.search(body)
            if m:
                ref.arxiv_id = re.sub(r"v\d+$", "", m.group(1))
                break

        refs.append(ref)

    return refs


def extract_cite_keys(tex_text: str) -> set[str]:
    """Extract all citation keys used in .tex files."""
    keys = set()
    for match in _CITE_CMD.finditer(tex_text):
        # \cite{key1,key2,key3}
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def extract_references(
    tex_files: list[Path],
    bib_files: list[Path],
    bbl_files: list[Path],
) -> list[Reference]:
    """Extract references from a paper's LaTeX source files.

    Priority: .bib (structured) > .bbl (semi-structured) > inline arxiv IDs.
    Only returns references that are actually \\cite'd in the .tex files.
    """
    # 1. Find all citation keys used in the paper
    all_tex = ""
    for tex_path in tex_files:
        try:
            all_tex += tex_path.read_text(errors="replace") + "\n"
        except Exception:
            continue

    cited_keys = extract_cite_keys(all_tex)
    logger.debug("Found %d citation keys in tex files", len(cited_keys))

    # 2. Parse .bib files (most structured)
    bib_refs: dict[str, Reference] = {}
    for bib_path in bib_files:
        try:
            bib_text = bib_path.read_text(errors="replace")
            for ref in parse_bib(bib_text):
                bib_refs[ref.cite_key] = ref
        except Exception as e:
            logger.debug("Failed to parse %s: %s", bib_path, e)

    # 3. Parse .bbl files (fallback for refs not in .bib)
    bbl_refs: dict[str, Reference] = {}
    for bbl_path in bbl_files:
        try:
            bbl_text = bbl_path.read_text(errors="replace")
            for ref in parse_bbl(bbl_text):
                bbl_refs[ref.cite_key] = ref
        except Exception as e:
            logger.debug("Failed to parse %s: %s", bbl_path, e)

    # 4. Merge: prefer .bib data, supplement with .bbl
    merged: dict[str, Reference] = {}
    all_keys = cited_keys | set(bib_refs.keys()) | set(bbl_refs.keys())

    for key in all_keys:
        if key in bib_refs:
            ref = bib_refs[key]
            # Supplement missing fields from .bbl
            if key in bbl_refs:
                bbl = bbl_refs[key]
                if not ref.title and bbl.raw_text:
                    ref.title = bbl.raw_text[:200]
                if not ref.year and bbl.year:
                    ref.year = bbl.year
                if not ref.arxiv_id and bbl.arxiv_id:
                    ref.arxiv_id = bbl.arxiv_id
            merged[key] = ref
        elif key in bbl_refs:
            merged[key] = bbl_refs[key]

    # 5. Filter to only cited references (if we found any cite keys)
    if cited_keys:
        result = [merged[k] for k in cited_keys if k in merged]
    else:
        # No cite keys found (maybe non-standard citation style), return all
        result = list(merged.values())

    logger.info(
        "Extracted %d references (%d from bib, %d from bbl, %d with arxiv IDs)",
        len(result),
        sum(1 for r in result if r.source == "bib"),
        sum(1 for r in result if r.source == "bbl"),
        sum(1 for r in result if r.arxiv_id),
    )
    return result
