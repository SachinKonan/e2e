"""CLI entry point for the paper crawler.

Usage:
    # Step 1: Crawl (downloads LaTeX source on-demand, needs metadata JSON)
    python -m paper_crawler crawl \
        --metadata ~/arxiv-metadata-oai-snapshot.json \
        --seeds 2401.13660 2312.10523 \
        --depth 3 --top-k 5

    # Quick stats on your cache
    python -m paper_crawler stats --cache-dir ./paper_cache

    # Pre-fetch specific papers without crawling
    python -m paper_crawler fetch --ids 2401.13660 2312.10523
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .cache import PaperCache
from .crawler import PaperCrawler
from .metadata import ArxivMetadata
from .ranker import ReferenceRanker
from .synth import SynthPipeline


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--cache-dir", default="./paper_cache", help="LaTeX source cache directory")
    parser.add_argument("-v", "--verbose", action="store_true")


def cmd_fetch(args: argparse.Namespace):
    """Pre-fetch LaTeX source for specific papers."""
    cache = PaperCache(args.cache_dir)
    ids = _load_ids(args)
    ok, fail = 0, 0
    for arxiv_id in ids:
        if cache.fetch(arxiv_id):
            ok += 1
        else:
            fail += 1
    print(f"Fetched: {ok} ok, {fail} failed/unavailable")


async def cmd_crawl_async(args: argparse.Namespace):
    """Run the BFS citation crawl."""
    seeds = _load_ids(args)

    # Load metadata
    metadata = ArxivMetadata()
    if args.metadata:
        metadata.load(args.metadata)
    else:
        print("Warning: no --metadata file, title resolution disabled", file=sys.stderr)

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
        metadata=metadata,
        ranker=ranker,
        synth=synth,
        top_k=args.top_k,
        max_depth=args.depth,
        max_papers=args.max_papers,
    )

    print(f"Crawling: {len(seeds)} seeds, depth={args.depth}, top_k={args.top_k}")
    if metadata:
        print(f"Metadata index: {len(metadata)} papers")
    result = await crawler.crawl(seeds)
    result.save(Path(args.output))

    if synth is not None:
        synth_path = Path(args.output).with_suffix(".synth.json")
        synth.store.save(synth_path)

    print(f"Done: {len(result.nodes)} papers, {len(result.skipped_no_source)} skipped")


def cmd_crawl(args: argparse.Namespace):
    asyncio.run(cmd_crawl_async(args))


def cmd_stats(args: argparse.Namespace):
    """Show cache statistics."""
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Cache dir {cache_dir} does not exist.")
        return

    total = 0
    ok = 0
    unavailable = 0
    for paper_dir in sorted(cache_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        total += 1
        status_file = paper_dir / "_status.json"
        if status_file.exists():
            import json
            status = json.loads(status_file.read_text())
            if status.get("status") == "ok":
                ok += 1
            else:
                unavailable += 1

    print(f"Cache: {total} papers ({ok} with source, {unavailable} unavailable)")


def _load_ids(args: argparse.Namespace) -> list[str]:
    ids = []
    if hasattr(args, "seeds") and args.seeds:
        ids.extend(args.seeds)
    if hasattr(args, "ids") and args.ids:
        ids.extend(args.ids)
    if hasattr(args, "seed_file") and args.seed_file and Path(args.seed_file).exists():
        with open(args.seed_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
    if not ids:
        print("Error: no paper IDs provided.", file=sys.stderr)
        sys.exit(1)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Paper crawler with LLM-ranked citations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- fetch ---
    p_fetch = subparsers.add_parser("fetch", help="Pre-fetch LaTeX source for specific papers")
    add_common_args(p_fetch)
    p_fetch.add_argument("--ids", nargs="+", help="Arxiv IDs to fetch")
    p_fetch.add_argument("--seed-file", type=Path, help="File with one arxiv ID per line")
    p_fetch.set_defaults(func=cmd_fetch)

    # --- crawl ---
    p_crawl = subparsers.add_parser("crawl", help="Run BFS citation crawl")
    add_common_args(p_crawl)
    p_crawl.add_argument("--metadata", type=Path, help="Path to arxiv-metadata-oai-snapshot.json from Kaggle")
    p_crawl.add_argument("--seeds", nargs="+", help="Arxiv IDs to start from")
    p_crawl.add_argument("--seed-file", type=Path, help="File with one arxiv ID per line")
    p_crawl.add_argument("--output", default="./crawl_result.json")
    p_crawl.add_argument("--depth", type=int, default=3)
    p_crawl.add_argument("--top-k", type=int, default=5)
    p_crawl.add_argument("--max-papers", type=int, default=500)
    p_crawl.add_argument("--ranker-url", default="http://localhost:8000/v1")
    p_crawl.add_argument("--ranker-model", default="default")
    p_crawl.add_argument("--ranker-api-key", default="not-needed")
    p_crawl.add_argument("--summarizer-url", default="http://localhost:8000/v1")
    p_crawl.add_argument("--summarizer-model", default="default")
    p_crawl.add_argument("--idea-url", default="http://localhost:8001/v1")
    p_crawl.add_argument("--idea-model", default="default")
    p_crawl.add_argument("--d-specific", action="store_true")
    p_crawl.add_argument("--no-synth", action="store_true")
    p_crawl.add_argument("--max-concurrent", type=int, default=16)
    p_crawl.set_defaults(func=cmd_crawl)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Show cache statistics")
    add_common_args(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
