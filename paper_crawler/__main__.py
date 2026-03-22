"""CLI entry point for the paper crawler.

Usage:
    # Step 1: Bulk download arxiv via NeMo Curator (one-time)
    python -m paper_crawler ingest --curator-dir /data/arxiv_curator_output

    # Step 1 (alt): Let Curator download directly
    python -m paper_crawler download --output-dir ./curator_raw --url-limit 10

    # Step 2: Crawl the citation graph
    python -m paper_crawler crawl --seeds 2401.13660 2312.10523 --depth 3 --top-k 5
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .cache import PaperCache
from .crawler import PaperCrawler
from .ranker import ReferenceRanker
from .synth import SynthPipeline


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--cache-dir", default="./paper_cache", help="Paper index directory")
    parser.add_argument("-v", "--verbose", action="store_true")


def cmd_download(args: argparse.Namespace):
    """Run NeMo Curator's arxiv bulk download and build the local index."""
    cache = PaperCache(args.cache_dir)
    cache.bulk_download(
        output_dir=args.output_dir,
        url_limit=args.url_limit,
        record_limit=args.record_limit,
    )
    print(f"Done. Index has {len(cache)} papers.")


def cmd_ingest(args: argparse.Namespace):
    """Ingest existing NeMo Curator output into the local index."""
    cache = PaperCache(args.cache_dir)
    cache.ingest_curator_output(args.curator_dir)
    cache.save_index()
    print(f"Done. Index has {len(cache)} papers.")


async def cmd_crawl_async(args: argparse.Namespace):
    """Run the BFS citation crawl."""
    seeds = _load_seeds(args)
    cache = PaperCache(args.cache_dir)
    cache.load_index()

    if len(cache) == 0:
        print("Error: paper index is empty. Run 'download' or 'ingest' first.", file=sys.stderr)
        sys.exit(1)

    # Check seed availability
    missing = [s for s in seeds if not cache.has(s)]
    if missing:
        print(f"Warning: {len(missing)} seed papers not in index: {missing[:5]}...", file=sys.stderr)
        seeds = [s for s in seeds if cache.has(s)]
        if not seeds:
            print("Error: no seed papers available in index.", file=sys.stderr)
            sys.exit(1)

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

    print(f"Crawling: {len(seeds)} seeds, depth={args.depth}, top_k={args.top_k}, index={len(cache)} papers")
    result = await crawler.crawl(seeds)
    result.save(Path(args.output))

    if synth is not None:
        synth_path = Path(args.output).with_suffix(".synth.json")
        synth.store.save(synth_path)

    print(f"Done: {len(result.nodes)} papers, {len(result.skipped_no_source)} skipped (no LaTeX source)")


def cmd_crawl(args: argparse.Namespace):
    asyncio.run(cmd_crawl_async(args))


def cmd_stats(args: argparse.Namespace):
    """Show index statistics."""
    cache = PaperCache(args.cache_dir)
    cache.load_index()
    print(f"Papers in index: {len(cache)}")
    if len(cache) > 0:
        ids = sorted(cache.available_ids())
        print(f"Sample IDs: {ids[:10]}")


def _load_seeds(args: argparse.Namespace) -> list[str]:
    seeds = []
    if args.seeds:
        seeds.extend(args.seeds)
    if args.seed_file and Path(args.seed_file).exists():
        with open(args.seed_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seeds.append(line)
    if not seeds:
        print("Error: no seed papers. Use --seeds or --seed-file.", file=sys.stderr)
        sys.exit(1)
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Paper crawler with LLM-ranked citations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    p_download = subparsers.add_parser("download", help="Bulk download arxiv via NeMo Curator")
    add_common_args(p_download)
    p_download.add_argument("--output-dir", default="./curator_raw", help="Where Curator writes raw output")
    p_download.add_argument("--url-limit", type=int, default=None, help="Max arxiv tar URLs to download")
    p_download.add_argument("--record-limit", type=int, default=None, help="Max records per tar file")
    p_download.set_defaults(func=cmd_download)

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest existing Curator JSONL output")
    add_common_args(p_ingest)
    p_ingest.add_argument("--curator-dir", required=True, help="Path to Curator JSONL output")
    p_ingest.set_defaults(func=cmd_ingest)

    # --- crawl ---
    p_crawl = subparsers.add_parser("crawl", help="Run BFS citation crawl")
    add_common_args(p_crawl)
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
    p_stats = subparsers.add_parser("stats", help="Show index statistics")
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
