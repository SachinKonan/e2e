"""LLM-based reference ranking using NeMo Curator's AsyncOpenAIClient.

Given a paper's context (title, abstract) and its references, asks an LLM
to rank references by relevance/importance, then returns the top-K.
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from nemo_curator import AsyncOpenAIClient

from .parser import ParsedPaper
from .references import Reference

logger = logging.getLogger(__name__)

RANK_PROMPT_TEMPLATE = """\
You are a research paper analyst. Given a source paper and its references, \
rank the references by their importance and relevance to the source paper's \
core contributions.

## Source Paper
Title: {title}
Abstract: {abstract}

## References
{references_block}

## Task
Rank ALL references by importance to the source paper. Return a JSON list of \
objects with fields "ref_number" and "reason" (one sentence), ordered from \
most to least important.

Return ONLY valid JSON. Example:
[
  {{"cite_key": "smith2023", "reason": "Core method this paper extends."}},
  {{"cite_key": "jones2024", "reason": "Provides the dataset used for evaluation."}}
]
"""


@dataclass
class RankedReference:
    """A reference with an LLM-assigned rank."""

    reference: Reference
    rank: int
    reason: str


def _format_references_block(references: list[Reference]) -> str:
    lines = []
    for i, ref in enumerate(references):
        parts = [f"[{ref.cite_key}]"]
        if ref.title:
            parts.append(ref.title)
        if ref.authors:
            parts.append(f"by {ref.authors[:100]}")
        if ref.year:
            parts.append(f"({ref.year})")
        if ref.arxiv_id:
            parts.append(f"[arxiv:{ref.arxiv_id}]")
        if not ref.title and ref.raw_text:
            parts.append(ref.raw_text[:200])
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _parse_ranking_response(response_text: str, references: list[Reference]) -> list[RankedReference]:
    """Parse the LLM's JSON ranking response."""
    text = response_text.strip()

    # Handle markdown code blocks
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        rankings = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM ranking response as JSON")
        return []

    # Build lookup by cite_key
    ref_by_key = {r.cite_key: r for r in references}

    ranked = []
    for rank_idx, entry in enumerate(rankings):
        cite_key = entry.get("cite_key", "")
        reason = entry.get("reason", "")
        if cite_key in ref_by_key:
            ranked.append(RankedReference(
                reference=ref_by_key[cite_key],
                rank=rank_idx + 1,
                reason=reason,
            ))

    return ranked


class ReferenceRanker:
    """Ranks paper references using an LLM via NeMo Curator."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "default",
        max_concurrent: int = 16,
    ):
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client = AsyncOpenAIClient(openai_client)
        self.model = model
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def rank_references(
        self,
        paper: ParsedPaper,
        top_k: int = 10,
        prompt_template: str | None = None,
    ) -> list[RankedReference]:
        """Rank a paper's references and return top-K.

        Args:
            paper: The parsed paper with references.
            top_k: Number of top references to return.
            prompt_template: Custom prompt template (uses default if None).

        Returns:
            Top-K ranked references, ordered by importance.
        """
        if not paper.references:
            logger.warning("No references to rank for %s", paper.arxiv_id)
            return []

        template = prompt_template or RANK_PROMPT_TEMPLATE
        prompt = template.format(
            title=paper.title or "Unknown",
            abstract=paper.abstract or "No abstract available.",
            references_block=_format_references_block(paper.references),
        )

        async with self._semaphore:
            response = await self.client.query_model(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.9,
                max_tokens=4096,
            )

        response_text = response["content"] if isinstance(response, dict) else str(response)
        ranked = _parse_ranking_response(response_text, paper.references)

        if not ranked:
            logger.warning("LLM returned no valid rankings for %s", paper.arxiv_id)
            return []

        logger.info(
            "Ranked %d/%d references for %s, returning top %d",
            len(ranked), len(paper.references), paper.arxiv_id, min(top_k, len(ranked)),
        )
        return ranked[:top_k]

    async def rank_batch(
        self,
        papers: list[ParsedPaper],
        top_k: int = 10,
    ) -> dict[str, list[RankedReference]]:
        """Rank references for multiple papers concurrently.

        Returns:
            Dict mapping arxiv_id -> list of top-K ranked references.
        """
        tasks = [self.rank_references(paper, top_k=top_k) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ranked_by_paper = {}
        for paper, result in zip(papers, results):
            if isinstance(result, Exception):
                logger.error("Ranking failed for %s: %s", paper.arxiv_id, result)
                ranked_by_paper[paper.arxiv_id] = []
            else:
                ranked_by_paper[paper.arxiv_id] = result

        return ranked_by_paper
