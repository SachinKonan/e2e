"""CLI entry point for the paper crawler.

Usage:
    python -m paper_crawler --seeds 2401.12345 2312.00001 --depth 3 --top-k 5
    python -m paper_crawler --seed-file seeds.txt --cache-dir ./papers
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .cache import PaperCache
from .config import PipelineConfig
from .crawler import PaperCrawler
from .ranker import ReferenceRanker
from .synth import SynthPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursive paper crawler with LLM-guided citation ranking",
    )
    parser.add_argument("--seeds", nargs="+", help="Arxiv IDs to start from")
    parser.add_argument("--seed-file", type=Path, help="File with one arxiv ID per line")
    parser.add_argument("--cache-dir", default="./paper_cache", help="Local PDF cache directory")
    parser.add_argument("--output", default="./crawl_result.json", help="Output JSON path")
    parser.add_argument("--depth", type=int, default=3, help="Max crawl depth")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K references to follow per paper")
    parser.add_argument("--max-papers", type=int, default=500, help="Max total papers to crawl")

    # LLM config for ranker
    parser.add_argument("--ranker-url", default="http://localhost:8000/v1", help="Ranker LLM base URL")
    parser.add_argument("--ranker-model", default="default", help="Ranker model name")
    parser.add_argument("--ranker-api-key", default="not-needed")

    # LLM config for synth (summarizer + idea extractor)
    parser.add_argument("--summarizer-url", default="http://localhost:8000/v1")
    parser.add_argument("--summarizer-model", default="default")
    parser.add_argument("--idea-url", default="http://localhost:8001/v1")
    parser.add_argument("--idea-model", default="default")
    parser.add_argument("--d-specific", action="store_true", help="Generate D-specific commentary (not reusable)")
    parser.add_argument("--no-synth", action="store_true", help="Skip summarization/idea extraction")

    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--config", type=Path, help="JSON config file (overrides CLI args)")
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args()


def load_seeds(args: argparse.Namespace) -> list[str]:
    seeds = []
    if args.seeds:
        seeds.extend(args.seeds)
    if args.seed_file and args.seed_file.exists():
        with open(args.seed_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seeds.append(line)
    if not seeds:
        print("Error: no seed papers provided. Use --seeds or --seed-file.", file=sys.stderr)
        sys.exit(1)
    return seeds


async def run(args: argparse.Namespace):
    seeds = load_seeds(args)

    cache = PaperCache(args.cache_dir)
    ranker = ReferenceRanker(
        base_url=args.ranker_url,
        api_key=args.ranker_api_key,
        model=args.ranker_model,
        max_concurrent=args.max_concurrent,
    )

    synth = None
    if not args.no_synth:
        synth = SynthPipeline(
            summarizer_base_url=args.summarizer_url,
            summarizer_model=args.summarizer_model,
            idea_base_url=args.idea_url,
            idea_model=args.idea_model,
            max_concurrent=args.max_concurrent,
            d_specific=args.d_specific,
        )

    crawler = PaperCrawler(
        cache=cache,
        ranker=ranker,
        synth=synth,
        top_k=args.top_k,
        max_depth=args.depth,
        max_papers=args.max_papers,
    )

    print(f"Starting crawl: {len(seeds)} seeds, depth={args.depth}, top_k={args.top_k}")
    result = await crawler.crawl(seeds)
    result.save(Path(args.output))

    # Save synth outputs alongside
    if synth is not None:
        synth_path = Path(args.output).with_suffix(".synth.json")
        synth.store.save(synth_path)

    print(f"Done: {len(result.nodes)} papers discovered. Output: {args.output}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
