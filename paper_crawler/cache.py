"""Local paper text cache backed by NeMo Curator's arxiv LaTeX extraction.

Two-tier design:
  1. Bulk preload: Use NeMo Curator's `download_arxiv()` to bulk-download arxiv
     tar bundles from S3, extract LaTeX → clean text, index by arxiv_id.
  2. Lookup: Crawler checks the index. Papers without LaTeX source are skipped
     (filter condition: must be on arxiv AND have LaTeX source available).

The index is a directory of JSONL files produced by NeMo Curator, plus a
fast in-memory dict mapping arxiv_id → extracted text.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperCache:
    """In-memory index over NeMo Curator's extracted arxiv LaTeX text."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # arxiv_id -> {"text": ..., "source_id": ..., "file_name": ...}
        self._index: dict[str, dict] = {}
        self._loaded = False

    def _index_path(self) -> Path:
        return self.cache_dir / "paper_index.json"

    def load_index(self):
        """Load the paper index from disk."""
        idx_path = self._index_path()
        if idx_path.exists():
            with open(idx_path) as f:
                self._index = json.load(f)
            logger.info("Loaded index with %d papers from %s", len(self._index), idx_path)
        self._loaded = True

    def save_index(self):
        """Persist the paper index to disk."""
        with open(self._index_path(), "w") as f:
            json.dump(self._index, f)
        logger.info("Saved index with %d papers to %s", len(self._index), self._index_path())

    def ingest_curator_output(self, curator_output_dir: str | Path):
        """Ingest JSONL files produced by NeMo Curator's download_arxiv().

        Each line in the JSONL has: {"text": ..., "id": ..., "source_id": ..., "file_name": ...}
        We index by the "id" field (arxiv ID).
        """
        curator_dir = Path(curator_output_dir)
        count = 0
        for jsonl_path in sorted(curator_dir.glob("*.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    arxiv_id = record.get("id", "")
                    if not arxiv_id:
                        continue
                    text = record.get("text", "")
                    if not text or len(text) < 100:
                        # Skip papers with essentially no extractable text
                        continue
                    self._index[arxiv_id] = {
                        "text": text,
                        "source_id": record.get("source_id", ""),
                        "file_name": record.get("file_name", ""),
                    }
                    count += 1
        logger.info("Ingested %d papers from %s (total index: %d)", count, curator_dir, len(self._index))
        self._loaded = True

    def bulk_download(
        self,
        output_dir: str | Path | None = None,
        url_limit: int | None = None,
        record_limit: int | None = None,
    ):
        """Run NeMo Curator's arxiv bulk download, then ingest the output.

        Requires s5cmd configured for arxiv S3 access.
        """
        from nemo_curator.download import download_arxiv

        out = Path(output_dir) if output_dir else self.cache_dir / "curator_raw"
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Starting NeMo Curator arxiv bulk download to %s", out)
        dataset = download_arxiv(
            output_path=str(out),
            output_type="jsonl",
            keep_raw_download=False,
            force_download=False,
            url_limit=url_limit,
            record_limit=record_limit,
        )
        # Write out the dataset
        dataset.to_json(output_path=str(out), write_to_filename=True)
        logger.info("Bulk download complete, ingesting...")
        self.ingest_curator_output(out)
        self.save_index()

    def has(self, arxiv_id: str) -> bool:
        """Check if paper text is available in the index."""
        if not self._loaded:
            self.load_index()
        return arxiv_id in self._index

    def get_text(self, arxiv_id: str) -> str | None:
        """Get extracted paper text by arxiv ID. Returns None if not available."""
        if not self._loaded:
            self.load_index()
        entry = self._index.get(arxiv_id)
        return entry["text"] if entry else None

    def get_metadata(self, arxiv_id: str) -> dict | None:
        """Get full cached record for a paper."""
        if not self._loaded:
            self.load_index()
        return self._index.get(arxiv_id)

    def available_ids(self) -> set[str]:
        """Return set of all arxiv IDs with LaTeX source available."""
        if not self._loaded:
            self.load_index()
        return set(self._index.keys())

    def __len__(self) -> int:
        if not self._loaded:
            self.load_index()
        return len(self._index)
