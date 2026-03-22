"""Recursive BFS paper crawler with LLM-guided citation pruning.

Algorithm:
  1. Start with seed papers (must exist in the NeMo Curator arxiv index)
  2. For each paper in the frontier:
     a. Look up text in local cache (LaTeX-extracted via NeMo Curator)
     b. Parse text, extract references with arxiv IDs
     c. Filter: only keep refs that ALSO exist in the cache (have LaTeX source)
     d. Send to LLM for ranking
     e. Take top-K ranked references
     f. Add to next frontier (if not already visited)
  3. Repeat for N levels

Key constraint: we only follow papers that are (1) on arxiv and (2) have
LaTeX source available in our NeMo Curator index. This naturally filters
the graph to high-quality, parseable papers.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .cache import PaperCache
from .parser import ParsedPaper, parse_paper
from .ranker import ReferenceRanker
from .synth import SynthPipeline

logger = logging.getLogger(__name__)


@dataclass
class CrawlNode:
    """A node in the citation graph."""

    arxiv_id: str
    depth: int
    parent_id: str | None = None
    rank_in_parent: int | None = None
    rank_reason: str | None = None
    parsed: ParsedPaper | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class CrawlResult:
    """Complete result of a crawl session."""

    nodes: dict[str, CrawlNode] = field(default_factory=dict)
    seed_ids: list[str] = field(default_factory=list)
    max_depth: int = 0
    skipped_no_source: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        result = {
            "seed_ids": self.seed_ids,
            "max_depth": self.max_depth,
            "total_papers": len(self.nodes),
            "skipped_no_source": len(self.skipped_no_source),
            "papers": {},
        }
        for aid, node in self.nodes.items():
            result["papers"][aid] = {
                "arxiv_id": node.arxiv_id,
                "depth": node.depth,
                "parent_id": node.parent_id,
                "rank_in_parent": node.rank_in_parent,
                "rank_reason": node.rank_reason,
                "title": node.parsed.title if node.parsed else None,
                "abstract": node.parsed.abstract if node.parsed else None,
                "num_references": len(node.parsed.references) if node.parsed else 0,
                "children": node.children,
            }
        return result

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved crawl result to %s (%d papers)", path, len(self.nodes))


class PaperCrawler:
    """BFS crawler that discovers papers via LLM-ranked citations."""

    def __init__(
        self,
        cache: PaperCache,
        ranker: ReferenceRanker,
        synth: SynthPipeline | None = None,
        top_k: int = 5,
        max_depth: int = 3,
        max_papers: int = 500,
    ):
        self.cache = cache
        self.ranker = ranker
        self.synth = synth
        self.top_k = top_k
        self.max_depth = max_depth
        self.max_papers = max_papers

    def _process_paper(self, arxiv_id: str) -> ParsedPaper | None:
        """Look up paper text in cache and parse it."""
        text = self.cache.get_text(arxiv_id)
        if text is None:
            logger.debug("No LaTeX source for %s, skipping", arxiv_id)
            return None

        try:
            parsed = parse_paper(arxiv_id, text)
            # Filter references to only those with LaTeX source available
            available = self.cache.available_ids()
            before = len(parsed.references)
            parsed.references = [
                r for r in parsed.references
                if r.arxiv_id and r.arxiv_id in available
            ]
            logger.info(
                "%s: %d refs total, %d with LaTeX source available",
                arxiv_id, before, len(parsed.references),
            )
            return parsed
        except Exception as e:
            logger.error("Failed to parse %s: %s", arxiv_id, e)
            return None

    async def crawl(self, seed_ids: list[str]) -> CrawlResult:
        """Run the recursive BFS crawl.

        Args:
            seed_ids: List of arxiv IDs to start from (must be in cache).

        Returns:
            CrawlResult containing the discovered citation graph.
        """
        result = CrawlResult(seed_ids=list(seed_ids), max_depth=self.max_depth)
        visited: set[str] = set()
        # (arxiv_id, depth, parent_id)
        frontier: list[tuple[str, int, str | None]] = [
            (aid, 0, None) for aid in seed_ids
        ]

        while frontier and len(result.nodes) < self.max_papers:
            current_level = frontier
            frontier = []
            depth = current_level[0][1] if current_level else 0

            logger.info(
                "=== Depth %d: %d papers to process (visited: %d, index size: %d) ===",
                depth, len(current_level), len(visited), len(self.cache),
            )

            papers_to_rank: list[tuple[CrawlNode, ParsedPaper]] = []

            for arxiv_id, d, parent_id in current_level:
                if arxiv_id in visited:
                    continue
                if len(result.nodes) >= self.max_papers:
                    break

                visited.add(arxiv_id)

                if not self.cache.has(arxiv_id):
                    result.skipped_no_source.add(arxiv_id)
                    continue

                parsed = self._process_paper(arxiv_id)
                if parsed is None:
                    continue

                node = CrawlNode(
                    arxiv_id=arxiv_id,
                    depth=d,
                    parent_id=parent_id,
                    parsed=parsed,
                )
                result.nodes[arxiv_id] = node

                # Fire off summarization + idea extraction
                if self.synth is not None:
                    asyncio.create_task(self.synth.process_paper(parsed))

                if d < self.max_depth and parsed.references:
                    papers_to_rank.append((node, parsed))

            if not papers_to_rank:
                continue

            # Rank references for all papers at this level
            parsed_list = [p for _, p in papers_to_rank]
            rankings = await self.ranker.rank_batch(parsed_list, top_k=self.top_k)

            # Build next frontier from top-K ranked references
            for node, parsed in papers_to_rank:
                ranked_refs = rankings.get(parsed.arxiv_id, [])
                for ranked in ranked_refs:
                    ref = ranked.reference
                    if ref.arxiv_id and ref.arxiv_id not in visited:
                        node.children.append(ref.arxiv_id)
                        frontier.append((ref.arxiv_id, node.depth + 1, node.arxiv_id))

                        if ref.arxiv_id not in result.nodes:
                            result.nodes[ref.arxiv_id] = CrawlNode(
                                arxiv_id=ref.arxiv_id,
                                depth=node.depth + 1,
                                parent_id=node.arxiv_id,
                                rank_in_parent=ranked.rank,
                                rank_reason=ranked.reason,
                            )

            logger.info(
                "Depth %d done: %d new papers queued, %d skipped (no LaTeX source)",
                depth, len(frontier), len(result.skipped_no_source),
            )

        logger.info(
            "Crawl complete: %d papers, %d skipped (no source), %d levels",
            len(result.nodes), len(result.skipped_no_source), self.max_depth,
        )
        return result
