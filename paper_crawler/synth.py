"""Summarization and idea extraction subroutines.

These run as side-effects during the crawl — as each paper is parsed,
we kick off async LLM calls to generate summaries and ideas.

Two modes:
  - Generic (d_specific=False): summaries/ideas are about the paper itself.
    These are REUSABLE across any citation path that includes this paper.
  - D-specific (d_specific=True): summaries/ideas are framed w.r.t. a target
    paper D. These are NOT reusable — each (paper, D) pair needs its own COT.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from nemo_curator import AsyncOpenAIClient

from .parser import ParsedPaper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARY_GENERIC_TEMPLATE = """\
Summarize the following research paper concisely. Focus on:
1. The core problem being addressed
2. The key methodology/approach
3. Main results and contributions

Title: {title}
Abstract: {abstract}

Full text (first 8000 chars):
{text_truncated}

Provide a 3-5 sentence summary.
"""

SUMMARY_D_SPECIFIC_TEMPLATE = """\
Summarize the following research paper, specifically highlighting aspects \
relevant to understanding paper D.

## Paper to summarize
Title: {title}
Abstract: {abstract}

Full text (first 8000 chars):
{text_truncated}

## Target paper D (context for why this paper matters)
Title: {d_title}
Abstract: {d_abstract}

Provide a 3-5 sentence summary focused on what this paper contributes \
toward understanding paper D.
"""

IDEAS_GENERIC_TEMPLATE = """\
Given this research paper, extract the key ideas and insights that would be \
valuable for a researcher building on this work.

Title: {title}
Abstract: {abstract}

Full text (first 8000 chars):
{text_truncated}

Return a JSON list of 3-7 ideas, each with fields "idea" (one sentence) \
and "type" (one of: "method", "finding", "limitation", "connection", "open_question").

Return ONLY valid JSON.
"""

IDEAS_D_SPECIFIC_TEMPLATE = """\
Given this research paper and a target paper D, extract key ideas from this \
paper that are specifically relevant to understanding or extending paper D.

## Source paper
Title: {title}
Abstract: {abstract}

Full text (first 8000 chars):
{text_truncated}

## Target paper D
Title: {d_title}
Abstract: {d_abstract}

Return a JSON list of 3-7 ideas, each with fields "idea" (one sentence) \
and "type" (one of: "method", "finding", "limitation", "connection", "open_question"). \
Focus on connections and relevance to paper D.

Return ONLY valid JSON.
"""


@dataclass
class PaperSummary:
    arxiv_id: str
    summary: str
    d_arxiv_id: str | None = None  # None means generic


@dataclass
class PaperIdeas:
    arxiv_id: str
    ideas: list[dict]  # [{"idea": ..., "type": ...}]
    d_arxiv_id: str | None = None  # None means generic


@dataclass
class SynthStore:
    """Accumulates generated summaries and ideas, with dedup."""

    summaries: dict[str, PaperSummary] = field(default_factory=dict)  # key: "{arxiv_id}" or "{arxiv_id}|{d_id}"
    ideas: dict[str, PaperIdeas] = field(default_factory=dict)

    def _key(self, arxiv_id: str, d_id: str | None) -> str:
        return f"{arxiv_id}|{d_id}" if d_id else arxiv_id

    def has_summary(self, arxiv_id: str, d_id: str | None = None) -> bool:
        return self._key(arxiv_id, d_id) in self.summaries

    def has_ideas(self, arxiv_id: str, d_id: str | None = None) -> bool:
        return self._key(arxiv_id, d_id) in self.ideas

    def add_summary(self, summary: PaperSummary):
        key = self._key(summary.arxiv_id, summary.d_arxiv_id)
        self.summaries[key] = summary

    def add_ideas(self, ideas: PaperIdeas):
        key = self._key(ideas.arxiv_id, ideas.d_arxiv_id)
        self.ideas[key] = ideas

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summaries": {
                k: {"arxiv_id": v.arxiv_id, "summary": v.summary, "d_arxiv_id": v.d_arxiv_id}
                for k, v in self.summaries.items()
            },
            "ideas": {
                k: {"arxiv_id": v.arxiv_id, "ideas": v.ideas, "d_arxiv_id": v.d_arxiv_id}
                for k, v in self.ideas.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d summaries, %d idea sets to %s", len(self.summaries), len(self.ideas), path)


class SynthPipeline:
    """Async pipeline for generating summaries and ideas as papers are crawled."""

    def __init__(
        self,
        summarizer_base_url: str = "http://localhost:8000/v1",
        summarizer_model: str = "default",
        idea_base_url: str = "http://localhost:8001/v1",
        idea_model: str = "default",
        api_key: str = "not-needed",
        max_concurrent: int = 16,
        d_specific: bool = False,
    ):
        self.summarizer = AsyncOpenAIClient(
            AsyncOpenAI(base_url=summarizer_base_url, api_key=api_key),
        )
        self.idea_client = AsyncOpenAIClient(
            AsyncOpenAI(base_url=idea_base_url, api_key=api_key),
        )
        self.summarizer_model = summarizer_model
        self.idea_model = idea_model
        self.d_specific = d_specific
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.store = SynthStore()

    async def summarize(
        self,
        paper: ParsedPaper,
        d_paper: ParsedPaper | None = None,
    ) -> PaperSummary | None:
        """Generate summary for a paper. Skips if already generated."""
        d_id = d_paper.arxiv_id if d_paper else None
        if self.store.has_summary(paper.arxiv_id, d_id):
            return self.store.summaries[self.store._key(paper.arxiv_id, d_id)]

        text_truncated = paper.full_text[:8000]

        if d_paper and self.d_specific:
            prompt = SUMMARY_D_SPECIFIC_TEMPLATE.format(
                title=paper.title or "Unknown",
                abstract=paper.abstract or "",
                text_truncated=text_truncated,
                d_title=d_paper.title or "Unknown",
                d_abstract=d_paper.abstract or "",
            )
        else:
            prompt = SUMMARY_GENERIC_TEMPLATE.format(
                title=paper.title or "Unknown",
                abstract=paper.abstract or "",
                text_truncated=text_truncated,
            )

        try:
            async with self._semaphore:
                response = await self.summarizer.query_model(
                    model=self.summarizer_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                )
            text = response["content"] if isinstance(response, dict) else str(response)
            summary = PaperSummary(arxiv_id=paper.arxiv_id, summary=text, d_arxiv_id=d_id)
            self.store.add_summary(summary)
            return summary
        except Exception as e:
            logger.error("Summary generation failed for %s: %s", paper.arxiv_id, e)
            return None

    async def extract_ideas(
        self,
        paper: ParsedPaper,
        d_paper: ParsedPaper | None = None,
    ) -> PaperIdeas | None:
        """Extract ideas from a paper. Skips if already generated."""
        d_id = d_paper.arxiv_id if d_paper else None
        if self.store.has_ideas(paper.arxiv_id, d_id):
            return self.store.ideas[self.store._key(paper.arxiv_id, d_id)]

        text_truncated = paper.full_text[:8000]

        if d_paper and self.d_specific:
            prompt = IDEAS_D_SPECIFIC_TEMPLATE.format(
                title=paper.title or "Unknown",
                abstract=paper.abstract or "",
                text_truncated=text_truncated,
                d_title=d_paper.title or "Unknown",
                d_abstract=d_paper.abstract or "",
            )
        else:
            prompt = IDEAS_GENERIC_TEMPLATE.format(
                title=paper.title or "Unknown",
                abstract=paper.abstract or "",
                text_truncated=text_truncated,
            )

        try:
            async with self._semaphore:
                response = await self.idea_client.query_model(
                    model=self.idea_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                )
            text = response["content"] if isinstance(response, dict) else str(response)

            # Parse JSON ideas
            try:
                if "```" in text:
                    start = text.find("[")
                    end = text.rfind("]") + 1
                    text = text[start:end]
                ideas_list = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                ideas_list = [{"idea": text, "type": "unknown"}]

            ideas = PaperIdeas(arxiv_id=paper.arxiv_id, ideas=ideas_list, d_arxiv_id=d_id)
            self.store.add_ideas(ideas)
            return ideas
        except Exception as e:
            logger.error("Idea extraction failed for %s: %s", paper.arxiv_id, e)
            return None

    async def process_paper(
        self,
        paper: ParsedPaper,
        d_paper: ParsedPaper | None = None,
    ) -> tuple[PaperSummary | None, PaperIdeas | None]:
        """Run both summarization and idea extraction concurrently."""
        summary_task = self.summarize(paper, d_paper)
        ideas_task = self.extract_ideas(paper, d_paper)
        summary, ideas = await asyncio.gather(summary_task, ideas_task)
        return summary, ideas
