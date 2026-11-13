"""
Run / index configuration.

config/config.yml  -> Config used to build an index
index_meta.json    -> Config saved next to each index (what actually produced it)
"""

from dataclasses import dataclass, asdict, fields
import json
import yaml


@dataclass
class Config:
    corpus: str
    chunking_strategy: str     # "naive" | "recursive"
    embedding_model: str       # "minilm" | "bge" | "gemini"
    k: int = 3                 # default top-k for queries / evaluation
    n_chunks: int = 0
    timestamp: str = ""


def dump_config(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)


def load_config(path):
    with open(path) as f:
        return Config(**yaml.safe_load(f))


def load_index_config(path):
    """Load an index_meta.json (JSON is valid YAML, so either loader works)."""
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in meta.items() if k in known})   # ignores e.g. legacy_meta
