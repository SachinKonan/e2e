"""Configuration for the paper crawler pipeline."""

from dataclasses import dataclass, field


@dataclass
class CacheConfig:
    cache_dir: str = "./paper_cache"


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    model: str = "default"
    max_concurrent: int = 16


@dataclass
class CrawlConfig:
    top_k: int = 5
    max_depth: int = 3
    max_papers: int = 500
    output_path: str = "./crawl_result.json"


@dataclass
class SynthConfig:
    """Config for summarization / idea extraction subroutines."""

    # Whether to generate D-specific commentary (can't reuse across paths)
    # or generic commentary (reusable across citation paths)
    d_specific: bool = False
    # LLM endpoint for summarization (can differ from ranker)
    summarizer_base_url: str = "http://localhost:8000/v1"
    summarizer_model: str = "default"
    # LLM endpoint for idea/COT generation (e.g., R1 for reasoning)
    idea_base_url: str = "http://localhost:8001/v1"
    idea_model: str = "default"
    max_concurrent: int = 16


@dataclass
class PipelineConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    ranker_llm: LLMConfig = field(default_factory=LLMConfig)
    crawl: CrawlConfig = field(default_factory=CrawlConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    seed_papers: list[str] = field(default_factory=list)
