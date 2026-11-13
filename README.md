# LegalBench-RAG — From-Scratch RAG Pipeline

**English** | [Français](README.fr.md)

A retrieval-augmented generation pipeline for the [LegalBench-RAG](https://arxiv.org/abs/2408.10343) benchmark, built from primitives — no RAG framework, no hosted vector DB. Retrieval is the rigorously evaluated core; generation is a grounded demonstration layer on top.

## Overview
- From-scratch RAG: ingest → chunk → embed → index → retrieve → generate
- Runs fully local/free (CPU) **or** with premium hosted embeddings (Gemini)
- Retrieval evaluated with LegalBench-RAG's own character-level metrics
- Controlled ablations across chunking strategy and embedding model
- With premium embeddings, **matches the published baseline** on PrivacyQA recall@8 (no reranker)
- Separated, persistent indexing and query pipelines
- Demonstrates: RAG construction, benchmark-faithful evaluation, controlled ablation, honest signal/noise interpretation

## Pipeline
- **Ingestion** — load LegalBench-RAG corpus + benchmark JSON
- **Chunking** — naive fixed-size (500 chars) **or** recursive character splitter + greedy merge
- **Embedding** — local (`MiniLM-L6-v2`, `BGE-small-en-v1.5`, CPU) **or** hosted (`gemini-embedding-2`)
- **Index** — persistent store (`embeddings.npy` + `chunks.json` + `index_meta.json`), built once, queried separately
- **Retrieval** — brute-force cosine similarity (NumPy), top-k
- **Generation** — hosted LLM over retrieved context with a grounded prompt *(demonstration layer; not yet evaluated)*
- **Evaluation** — character-level Recall@k, Precision@k, MRR

## Metrics
- **Recall@k** — fraction of gold snippet characters covered by retrieved chunks
- **Precision@k** — fraction of retrieved characters that are gold
- **MRR** — rank of the first relevant chunk
- Character-level scoring = LegalBench-RAG's deterministic methodology

## Results
Subset: LegalBench-RAG-mini · 194 questions/corpus.

**Chunking ablation** · `MiniLM-L6-v2` (local) · k=3

| Corpus | Chunking | Recall | Precision | MRR |
|---|---|---|---|---|
| ContractNLI | naive | 0.071 | 0.020 | 0.103 |
| ContractNLI | recursive | **0.097** | **0.029** | 0.102 |
| PrivacyQA | naive | 0.124 | 0.073 | 0.205 |
| PrivacyQA | recursive | 0.120 | **0.086** | **0.227** |

**Embedding-model ablation** · PrivacyQA · recursive · k=3

| Model | Recall | Precision | MRR |
|---|---|---|---|
| MiniLM-L6-v2 (local) | 0.120 | 0.086 | 0.227 |
| BGE-small-en-v1.5 (local) | 0.120 | 0.070 | 0.182 |
| gemini-embedding-2 (hosted) | **0.264** | **0.180** | **0.441** |

**vs. published baseline** · PrivacyQA · recursive · no reranker · k=8

| Setup | Recall | Precision |
|---|---|---|
| This project (`gemini-embedding-2`) | 0.414 | 0.118 |
| Paper (OpenAI `text-embedding-3-large`) | 0.424 | 0.090 |

## Findings
- **Embedding model is the dominant lever:** local/free ≈ 0.12 recall@3; premium (Gemini) ≈ 0.26 — roughly 2×
- With premium embeddings, retrieval **matches the paper's reference baseline** on PrivacyQA recall@8 (0.414 vs 0.424, no reranker); precision is higher (0.118 vs 0.090), consistent with using smaller chunks
- Recursive chunking improved ContractNLI (structured contracts); PrivacyQA within noise
- BGE-small did not beat MiniLM at this scale — differences within noise
- N=194: differences under ~0.04 are within noise — reported as such, not over-read

## Limitations
- `gemini-embedding-2` and generation use **hosted APIs** (not free/local in that configuration)
- Generation is a demonstration layer — **not yet evaluated** (faithfulness/relevance)
- Gemini embedding run on PrivacyQA only so far
- Simplified recursive splitter (flatten-then-merge, no overlap); not LangChain's RCTS
- Chunk size (500 chars) differs from the paper's unstated RCTS size — precision not directly comparable
- Brute-force retrieval (fine at this scale, not production)
- Evaluation subset: mini, 194 Q/corpus, ≤2 of 4 corpora

## Stack
`Python` · `sentence-transformers` · `google-genai` · `NumPy` · `pandas`

## Structure
```
app.py                     Streamlit UI: ask, compare models side by side
config/config.yml          index build config (corpus, chunking, embedding model, k)
src/                       library
  data_ingesting.py        corpus file listing
  chunking.py              naive + recursive chunkers
  embedding.py             embedders (MiniLM / BGE local, Gemini hosted)
  retrieval.py             cosine top-k (NumPy)
  pipeline.py              build_index() + Index (load once, query many)
  generate_answer.py       grounded generation over retrieved chunks
  eval_metrics.py          character-level Recall / Precision / MRR
scripts/                   entry points (run from repo root with `python -m`)
  build_index.py           build an index from the config
  evaluate.py              evaluate a saved index on a benchmark
  make_mini_benchmark.py   sample the 194-question *_mini.json subsets
data/legalBench-RAG/       corpus + benchmarks
indexes/<corpus>/<ts>/     chunks.json, embeddings.npy, index_meta.json
results/<corpus>/          evaluation runs (JSON)
```

## Usage
```bash
pip install -r requirements.txt
cp .env.example .env                    # add GEMINI_API_KEY (only for gemini / generation)
python -m scripts.build_index --corpus privacy_qa --embedding-model minilm
python -m scripts.evaluate indexes/privacy_qa/<timestamp> --k 3
streamlit run app.py                    # interactive: ask a question, compare models
```

## Reference
Pipitone & Houir Alami (2024), *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain*. [arXiv:2408.10343](https://arxiv.org/abs/2408.10343)

