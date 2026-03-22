"""Local LaTeX source cache with per-paper arxiv download.

OSM tile-cache pattern: check local first, fetch on miss.
Downloads individual LaTeX source tarballs from arxiv.org/src/{id},
extracts .tex and .bib/.bbl files, and caches the results.

Directory layout:
  {cache_dir}/
    {arxiv_id}/           # e.g., 2401.12345/
      raw.tar.gz          # original source tarball
      *.tex               # extracted tex files
      *.bib               # bibtex files (if present)
      *.bbl               # compiled bibliography (if present)
      _text.txt           # concatenated body text (extracted from .tex)
      _status.json        # download status + metadata
"""

import gzip
import io
import json
import logging
import re
import tarfile
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ARXIV_SRC_URL = "https://arxiv.org/src/{arxiv_id}"
FETCH_DELAY_S = 3.0
MAX_RETRIES = 3

# Basic LaTeX stripping patterns
_LATEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_LATEX_COMMAND = re.compile(r"\\(?:begin|end)\{[^}]*\}")
_LATEX_MACRO = re.compile(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*")
_LATEX_BRACES = re.compile(r"[{}]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r" {2,}")


def _strip_latex(tex: str) -> str:
    """Lightweight LaTeX → plain text conversion."""
    text = _LATEX_COMMENT.sub("", tex)
    text = _LATEX_COMMAND.sub("", text)
    # Keep content of common text commands
    for cmd in ("textbf", "textit", "emph", "text", "mathrm"):
        text = re.sub(rf"\\{cmd}\{{([^}}]*)\}}", r"\1", text)
    text = _LATEX_MACRO.sub(" ", text)
    text = _LATEX_BRACES.sub("", text)
    text = text.replace("~", " ").replace("\\\\", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


class PaperCache:
    """Filesystem-backed LaTeX source cache with arxiv auto-download."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_fetch_time: float = 0.0
        # Track papers we know have no source available
        self._unavailable: set[str] = set()

    def _paper_dir(self, arxiv_id: str) -> Path:
        safe_id = arxiv_id.replace("/", "_")
        return self.cache_dir / safe_id

    def _status_path(self, arxiv_id: str) -> Path:
        return self._paper_dir(arxiv_id) / "_status.json"

    def _text_path(self, arxiv_id: str) -> Path:
        return self._paper_dir(arxiv_id) / "_text.txt"

    def has(self, arxiv_id: str) -> bool:
        """Check if LaTeX source is cached locally."""
        return self._text_path(arxiv_id).exists()

    def is_unavailable(self, arxiv_id: str) -> bool:
        """Check if we already know this paper has no LaTeX source."""
        if arxiv_id in self._unavailable:
            return True
        status_path = self._status_path(arxiv_id)
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") == "unavailable":
                self._unavailable.add(arxiv_id)
                return True
        return False

    def get_text(self, arxiv_id: str) -> str | None:
        """Get extracted plain text for a paper."""
        text_path = self._text_path(arxiv_id)
        if text_path.exists():
            return text_path.read_text()
        return None

    def get_tex_files(self, arxiv_id: str) -> list[Path]:
        """Get all .tex files for a paper."""
        paper_dir = self._paper_dir(arxiv_id)
        if not paper_dir.exists():
            return []
        return sorted(paper_dir.glob("*.tex"))

    def get_bib_files(self, arxiv_id: str) -> list[Path]:
        """Get all .bib files for a paper."""
        paper_dir = self._paper_dir(arxiv_id)
        if not paper_dir.exists():
            return []
        return sorted(paper_dir.glob("*.bib"))

    def get_bbl_files(self, arxiv_id: str) -> list[Path]:
        """Get all .bbl files (compiled bibliography) for a paper."""
        paper_dir = self._paper_dir(arxiv_id)
        if not paper_dir.exists():
            return []
        return sorted(paper_dir.glob("*.bbl"))

    def _rate_limit(self):
        elapsed = time.time() - self._last_fetch_time
        if elapsed < FETCH_DELAY_S:
            time.sleep(FETCH_DELAY_S - elapsed)

    def _mark_unavailable(self, arxiv_id: str, reason: str):
        paper_dir = self._paper_dir(arxiv_id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        status = {"status": "unavailable", "reason": reason}
        self._status_path(arxiv_id).write_text(json.dumps(status))
        self._unavailable.add(arxiv_id)

    def fetch(self, arxiv_id: str, force: bool = False) -> bool:
        """Download LaTeX source from arxiv, extract, and cache.

        Args:
            arxiv_id: e.g. "2401.12345"
            force: re-download even if cached

        Returns:
            True if source was successfully fetched and extracted.
        """
        if not force and self.has(arxiv_id):
            return True
        if not force and self.is_unavailable(arxiv_id):
            return False

        paper_dir = self._paper_dir(arxiv_id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        url = ARXIV_SRC_URL.format(arxiv_id=arxiv_id)

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "paper-crawler/0.1"})
                with urllib.request.urlopen(req) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    data = resp.read()
                    self._last_fetch_time = time.time()

                    # arxiv returns application/x-eprint-tar for LaTeX source
                    # or application/pdf if only PDF is available
                    if "pdf" in content_type.lower():
                        self._mark_unavailable(arxiv_id, "pdf-only")
                        logger.info("%s: PDF-only, no LaTeX source", arxiv_id)
                        return False

                    # Try to extract as tar.gz
                    return self._extract_tarball(arxiv_id, data, paper_dir)

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    self._mark_unavailable(arxiv_id, "not-found")
                    logger.info("%s: not found on arxiv", arxiv_id)
                    return False
                logger.warning("Attempt %d/%d for %s: HTTP %d", attempt, MAX_RETRIES, arxiv_id, e.code)
            except Exception as e:
                logger.warning("Attempt %d/%d for %s: %s", attempt, MAX_RETRIES, arxiv_id, e)

            time.sleep(2**attempt)

        self._mark_unavailable(arxiv_id, "download-failed")
        return False

    def _extract_tarball(self, arxiv_id: str, data: bytes, paper_dir: Path) -> bool:
        """Extract .tex, .bib, .bbl files from a source tarball."""
        # Save raw tarball
        raw_path = paper_dir / "raw.tar.gz"
        raw_path.write_bytes(data)

        extracted_any = False
        tex_contents = []

        try:
            # Try as gzipped tar first
            fileobj = io.BytesIO(data)
            try:
                tar = tarfile.open(fileobj=fileobj, mode="r:gz")
            except tarfile.ReadError:
                # Maybe it's just gzipped tex (single file, not a tar)
                try:
                    decompressed = gzip.decompress(data)
                    # Single .tex file
                    tex_path = paper_dir / "main.tex"
                    tex_path.write_bytes(decompressed)
                    tex_contents.append(decompressed.decode("utf-8", errors="replace"))
                    extracted_any = True
                    tar = None
                except Exception:
                    logger.warning("%s: could not decompress source", arxiv_id)
                    self._mark_unavailable(arxiv_id, "unreadable-archive")
                    return False

            if tar is not None:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = Path(member.name).name.lower()
                    ext = Path(member.name).suffix.lower()

                    if ext in (".tex", ".bib", ".bbl"):
                        # Extract to paper_dir with flat names
                        safe_name = Path(member.name).name
                        target = paper_dir / safe_name
                        try:
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            content = f.read()
                            target.write_bytes(content)
                            extracted_any = True

                            if ext == ".tex":
                                tex_contents.append(content.decode("utf-8", errors="replace"))
                        except Exception as e:
                            logger.debug("Failed to extract %s from %s: %s", member.name, arxiv_id, e)
                tar.close()

        except Exception as e:
            logger.warning("%s: failed to extract tarball: %s", arxiv_id, e)
            self._mark_unavailable(arxiv_id, "extraction-failed")
            return False

        if not extracted_any:
            self._mark_unavailable(arxiv_id, "no-tex-files")
            return False

        # Generate plain text from .tex files
        combined_tex = "\n\n".join(tex_contents)
        plain_text = _strip_latex(combined_tex)
        self._text_path(arxiv_id).write_text(plain_text)

        # Status
        status = {
            "status": "ok",
            "tex_files": [p.name for p in paper_dir.glob("*.tex")],
            "bib_files": [p.name for p in paper_dir.glob("*.bib")],
            "bbl_files": [p.name for p in paper_dir.glob("*.bbl")],
            "text_length": len(plain_text),
        }
        self._status_path(arxiv_id).write_text(json.dumps(status))

        logger.info(
            "%s: extracted %d tex, %d bib, %d bbl files (%d chars text)",
            arxiv_id,
            len(status["tex_files"]),
            len(status["bib_files"]),
            len(status["bbl_files"]),
            len(plain_text),
        )
        return True
