# LegalBench-RAG — From-Scratch Retrieval Pipeline

**English** | [Français](README.fr.md)

A retrieval pipeline for the [LegalBench-RAG](https://arxiv.org/abs/2408.10343) benchmark, built from primitives — no RAG framework, no hosted vector DB. Focus: the retrieval stage and its evaluation.

## Overview
- From-scratch RAG retrieval: ingest → chunk → embed → retrieve → evaluate
- Free / local components only (CPU, no GPU, no paid API)
- Evaluated with LegalBench-RAG's own character-level metrics
- Two chunking strategies compared via controlled ablation
- Demonstrates: retrieval-system construction, benchmark-faithful evaluation, controlled ablation, honest signal/noise interpretation

## Pipeline
- **Ingestion** — load LegalBench-RAG corpus + benchmark JSON
- **Chunking** — (1) naive fixed-size (500 chars); (2) recursive character splitter + greedy merge
- **Embedding** — `sentence-transformers`, local CPU (MiniLM-L6-v2, BGE-small-en-v1.5)
- **Retrieval** — brute-force cosine similarity (NumPy), top-k
- **Evaluation** — character-level Recall@k, Precision@k, MRR

## Metrics
- **Recall@k** — fraction of gold snippet characters covered by retrieved chunks
- **Precision@k** — fraction of retrieved characters that are gold
- **MRR** — rank of the first relevant chunk
- Character-level scoring = LegalBench-RAG's deterministic methodology

## Results
Subset: LegalBench-RAG-mini · 194 questions/corpus · k=3 · MiniLM-L6-v2

| Corpus | Chunking | Recall | Precision | MRR |
|---|---|---|---|---|
| ContractNLI | naive | 0.071 | 0.020 | 0.103 |
| ContractNLI | recursive | **0.097** | **0.029** | 0.102 |
| PrivacyQA | naive | 0.124 | 0.073 | 0.205 |
| PrivacyQA | recursive | 0.120 | **0.086** | **0.227** |

## Findings
- Recursive chunking improved ContractNLI (structured contracts); PrivacyQA within noise
- Precision improved with recursive chunking on both corpora
- BGE-small (free) did not robustly beat MiniLM at this scale — differences within noise
- Absolute scores are far below the paper's reference setup (OpenAI `text-embedding-3-large` + Cohere reranker); embedding-model quality is the dominant gap
- N=194: differences under ~0.04 are within noise — reported as such, not over-read

## Limitations
- Evaluation subset (mini, 194 Q/corpus, 2 of 4 corpora)
- Simplified recursive splitter (flatten-then-merge, no overlap)
- Brute-force retrieval (fine at this scale, not production)
- Free CPU embedding models only

## Stack
`Python` · `sentence-transformers` · `NumPy` · `pandas`

## Structure
- `data_ingesting.py` — corpus + benchmark loading
- `chunking.py` — naive + recursive chunkers
- `embedding.py` — embedding models
- `retrieval.py` — cosine retrieval + top-k
- `retrieval_eval_metrics.py` — Recall / Precision / MRR
- `main.py` — pipeline orchestration

## Reference
Pipitone & Houir Alami (2024), *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain*. [arXiv:2408.10343](https://arxiv.org/abs/2408.10343)