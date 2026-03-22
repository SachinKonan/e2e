"""Local PDF cache with arxiv download support.

Mirrors the OSM tile-cache pattern: check local path first, fetch on miss.
PDFs are stored at: {cache_dir}/{arxiv_id}.pdf
where arxiv_id has slashes replaced with underscores (e.g., 2401.12345 -> 2401.12345.pdf).
"""

import logging
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
# arxiv rate limit: be polite
FETCH_DELAY_S = 3.0
MAX_RETRIES = 3


class PaperCache:
    """Filesystem-backed PDF cache with arxiv auto-download."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_fetch_time: float = 0.0

    def _pdf_path(self, arxiv_id: str) -> Path:
        safe_id = arxiv_id.replace("/", "_")
        return self.cache_dir / f"{safe_id}.pdf"

    def has(self, arxiv_id: str) -> bool:
        """Check if PDF is already cached locally."""
        path = self._pdf_path(arxiv_id)
        return path.exists() and path.stat().st_size > 0

    def get_path(self, arxiv_id: str) -> Path | None:
        """Return local path if cached, else None."""
        if self.has(arxiv_id):
            return self._pdf_path(arxiv_id)
        return None

    def _rate_limit(self):
        """Respect arxiv rate limits."""
        elapsed = time.time() - self._last_fetch_time
        if elapsed < FETCH_DELAY_S:
            time.sleep(FETCH_DELAY_S - elapsed)

    def fetch(self, arxiv_id: str, force: bool = False) -> Path:
        """Download PDF from arxiv if not cached. Returns local path.

        Args:
            arxiv_id: e.g. "2401.12345" or "cs/0601001"
            force: re-download even if cached

        Returns:
            Path to the local PDF file.

        Raises:
            RuntimeError: if download fails after retries.
        """
        path = self._pdf_path(arxiv_id)
        if not force and self.has(arxiv_id):
            logger.debug("Cache hit: %s", arxiv_id)
            return path

        url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        logger.info("Downloading %s -> %s", url, path)

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                urllib.request.urlretrieve(url, path)
                self._last_fetch_time = time.time()
                if path.stat().st_size > 0:
                    logger.info("Downloaded %s (%d bytes)", arxiv_id, path.stat().st_size)
                    return path
                else:
                    logger.warning("Empty file for %s, retrying", arxiv_id)
                    path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, arxiv_id, e)
                backoff = 2**attempt
                time.sleep(backoff)

        raise RuntimeError(f"Failed to download {arxiv_id} after {MAX_RETRIES} attempts")

    def fetch_batch(self, arxiv_ids: list[str], force: bool = False) -> dict[str, Path]:
        """Download multiple papers, skipping already-cached ones.

        Returns:
            Dict mapping arxiv_id -> local path for successfully fetched papers.
        """
        results = {}
        skipped = 0
        for arxiv_id in arxiv_ids:
            if not force and self.has(arxiv_id):
                results[arxiv_id] = self._pdf_path(arxiv_id)
                skipped += 1
                continue
            try:
                results[arxiv_id] = self.fetch(arxiv_id, force=force)
            except RuntimeError:
                logger.error("Skipping %s after download failure", arxiv_id)
        logger.info("Batch complete: %d fetched, %d cached, %d failed", len(results) - skipped, skipped, len(arxiv_ids) - len(results))
        return results
