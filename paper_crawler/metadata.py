"""Arxiv metadata index from the Kaggle Cornell University dataset.

Dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
Format: One JSON object per line with fields:
  id, submitter, authors, title, comments, journal-ref, doi, abstract,
  categories, versions

This gives us title/abstract/categories for ~1.7M papers, enabling:
  - Resolve reference titles → arxiv IDs
  - Pre-filter seeds by category (cs.LG, cs.CL, etc.)
  - Provide title/abstract context without downloading source
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ArxivMetadata:
    """In-memory index over the Kaggle arxiv metadata JSON."""

    def __init__(self):
        # arxiv_id -> {title, authors, abstract, categories, ...}
        self._by_id: dict[str, dict] = {}
        # normalized title -> arxiv_id (for resolving references by title)
        self._by_title: dict[str, str] = {}
        self._loaded = False

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for fuzzy matching."""
        return " ".join(title.lower().strip().split())

    def load(self, metadata_path: str | Path):
        """Load the Kaggle arxiv-metadata-oai-snapshot.json file.

        This is a line-delimited JSON file (~3.5GB), one record per line.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        logger.info("Loading arxiv metadata from %s ...", metadata_path)
        count = 0
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                arxiv_id = record.get("id", "")
                if not arxiv_id:
                    continue

                entry = {
                    "id": arxiv_id,
                    "title": record.get("title", "").replace("\n", " ").strip(),
                    "authors": record.get("authors", ""),
                    "abstract": record.get("abstract", "").replace("\n", " ").strip(),
                    "categories": record.get("categories", ""),
                    "doi": record.get("doi", ""),
                    "journal_ref": record.get("journal-ref", ""),
                    "versions": record.get("versions", []),
                }
                self._by_id[arxiv_id] = entry

                # Index by normalized title for reference resolution
                norm_title = self._normalize_title(entry["title"])
                if norm_title and len(norm_title) > 10:
                    self._by_title[norm_title] = arxiv_id

                count += 1
                if count % 500_000 == 0:
                    logger.info("  ... loaded %d papers", count)

        self._loaded = True
        logger.info("Loaded %d papers (%d unique titles)", len(self._by_id), len(self._by_title))

    def get(self, arxiv_id: str) -> dict | None:
        """Look up metadata by arxiv ID."""
        return self._by_id.get(arxiv_id)

    def resolve_title(self, title: str) -> str | None:
        """Try to find an arxiv ID for a given paper title.

        Returns arxiv_id if found, None otherwise.
        """
        norm = self._normalize_title(title)
        return self._by_title.get(norm)

    def has(self, arxiv_id: str) -> bool:
        return arxiv_id in self._by_id

    def filter_by_categories(self, categories: set[str]) -> list[str]:
        """Return arxiv IDs that match any of the given categories.

        Useful for building seed lists (e.g., all cs.LG, cs.CL papers).
        """
        results = []
        for arxiv_id, entry in self._by_id.items():
            paper_cats = set(entry["categories"].split())
            if paper_cats & categories:
                results.append(arxiv_id)
        return results

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, arxiv_id: str) -> bool:
        return arxiv_id in self._by_id
