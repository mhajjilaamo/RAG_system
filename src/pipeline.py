"""
Indexing and query pipelines (formerly src/main.py).

Indexing:  corpus -> chunks -> embeddings -> indexes/<corpus>/<timestamp>/
Query:     question -> embedding -> cosine top-k over a saved index -> chunks
"""
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from src.chunking import Chunk, fixed_size_chunker, recursive_chunker
from src.config import Config, dump_config, load_config, load_index_config
from src.data_ingesting import import_data_legalBench_RAG_corpus
from src.embedding import get_embedder
from src.retrieval import retrieve_answer, retrieve_with_scores

CHUNKERS = {"naive": fixed_size_chunker, "recursive": recursive_chunker}
INDEXES_ROOT = "indexes"


def print_pretty_results(question, results):
    print("Question :", question, "\n")
    for i, result in enumerate(results):
        print("Top answer n", i, " :", result, "\n")


# ---------------------------------------------------------------- indexing
def build_index(config: Config, indexes_root=INDEXES_ROOT):
    """Chunk + embed a whole corpus and persist it. Returns the index directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    index_dir = os.path.join(indexes_root, config.corpus, timestamp)
    os.makedirs(index_dir, exist_ok=True)
    print("building index :", index_dir)

    # Load + chunk corpus
    chunker = CHUNKERS[config.chunking_strategy]
    chunks = []
    for file in import_data_legalBench_RAG_corpus(config.corpus):
        with open(file.path_to_corpus + file.file, "r", encoding="utf-8") as f:
            chunks.extend(chunker(f.read(), file.file))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=4)
    print(f"chunks done ({len(chunks)})")

    # Embed chunks
    embed_docs, _ = get_embedder(config)
    embeddings = np.array(embed_docs([c.text for c in chunks]), dtype=np.float32)
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
    print("embedding chunks done")

    # Metadata: the config that produced this index
    config.timestamp = timestamp
    config.n_chunks = len(chunks)
    dump_config(os.path.join(index_dir, "index_meta.json"), config)
    return index_dir


# ---------------------------------------------------------------- querying
class Index:
    """A saved index loaded once: chunks, embeddings, and the matching query embedder.

    The query embedder is chosen from the index's own index_meta.json, so a question
    is always embedded with the same model as the chunks."""

    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.config = load_index_config(os.path.join(index_dir, "index_meta.json"))
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            self.chunks = [Chunk(**d) for d in json.load(f)]
        _, self.embed_query = get_embedder(self.config)

    def query(self, question, n_results=3):
        q = np.array(self.embed_query(question), dtype=np.float32)
        return [self.chunks[i] for i in retrieve_answer(q, self.embeddings, n_results)]

    def query_with_scores(self, question, n_results=3):
        """Like query(), but returns [(chunk, cosine_similarity), ...]."""
        q = np.array(self.embed_query(question), dtype=np.float32)
        return [(self.chunks[i], s) for i, s in retrieve_with_scores(q, self.embeddings, n_results)]


def ask_vector_store_pipeline(question, index_dir, n_results=3):
    """One-off query (loads the index each call — use Index directly in loops)."""
    results = Index(index_dir).query(question, n_results)
    print_pretty_results(question, results)
    return question, results
