# LegalBench-RAG — From-Scratch RAG

RAG pipeline for [LegalBench-RAG](https://arxiv.org/abs/2408.10343), built from primitives — no RAG framework, no vector DB.

`ingest → chunk → embed → index → retrieve → generate → evaluate`

| Stage | Options |
|---|---|
| Chunking | `naive` (fixed 500 chars) · `recursive` (split on paragraphs/lines/sentences, merge ≤ 500 chars) |
| Embedding | `minilm` (all-MiniLM-L6-v2, local) · `bge` (bge-small-en-v1.5, local) · `gemini` (gemini-embedding-2, API) |
| Retrieval | brute-force cosine top-k (NumPy) |
| Generation | Gemini over retrieved chunks (demo, not evaluated) |
| Metrics | character-level Recall@k · Precision@k · MRR (LegalBench-RAG method) |

---

## Setup

Requirements: Python 3.10+, a few GB of disk (PyTorch). GPU not needed.

```bash
git clone git@github.com:mhajjilaamo/RAG_system.git
cd RAG_system
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional — Gemini (embeddings and answer generation only):

```bash
cp .env.example .env        # then set GEMINI_API_KEY=<your key>
```

Data: corpus + benchmarks are in `data/legalBench-RAG/` (194-question `*_mini.json` subsets included).
Local models download from Hugging Face on first use.

> Run every command from the repo root.

---

## Run

| Task | Command |
|---|---|
| Web app | `streamlit run app.py` → http://localhost:8501 |
| Build an index | `python -m scripts.build_index --corpus privacy_qa --chunking-strategy naive --embedding-model minilm` |
| Evaluate an index | `python -m scripts.evaluate indexes/privacy_qa/<timestamp> --k 3` |
| Rebuild mini benchmarks | `python -m scripts.make_mini_benchmark` |

`build_index` flags (default: `config/config.yml`):
- `--corpus` `contractnli` · `cuad` · `maud` · `privacy_qa`
- `--chunking-strategy` `naive` · `recursive`
- `--embedding-model` `minilm` · `bge` · `gemini`

`evaluate` flags: `--k N` · `--max-questions N` · `--full` (full benchmark, not mini) · `--verbose`

Outputs:
- indexes → `indexes/<corpus>/<timestamp>/` (`chunks.json`, `embeddings.npy`, `index_meta.json`)
- eval runs → `results/<corpus>/run_<corpus>_<timestamp>.json`

---

## Use the app

1. **Sidebar** — pick corpus, embedding models, chunking strategies, `k`.
   Every selected model × chunking combo that has an index is compared side by side.
2. **Question** — choose a benchmark question (gold answer shown, metrics computed) or type your own.
3. **Ask** — see:
   - comparison table: retrieval time, cosine scores, recall / precision / RR, gold hits
   - ranked chunks per configuration (✅ = overlaps gold answer)
4. **Generate answer** toggle — Gemini answer per configuration (needs `.env`; 60 s timeout).
5. **Build an index** (sidebar) — create a new local index (MiniLM / BGE) in one click.
6. **Benchmark runs** tab — saved evaluation results for the corpus.

Notes:
- Cosine scores are not comparable across models.
- Gemini indexes: build from CLI (free tier pauses 60 s per 100 chunks).

---

## Results

LegalBench-RAG mini · 194 questions/corpus.

**Chunking** · MiniLM · k=3

| Corpus | Chunking | Recall | Precision | MRR |
|---|---|---|---|---|
| ContractNLI | naive | 0.071 | 0.020 | 0.103 |
| ContractNLI | recursive | **0.097** | **0.029** | 0.102 |
| PrivacyQA | naive | 0.124 | 0.073 | 0.205 |
| PrivacyQA | recursive | 0.120 | **0.086** | **0.227** |

**Embedding model** · PrivacyQA · recursive · k=3

| Model | Recall | Precision | MRR |
|---|---|---|---|
| MiniLM | 0.120 | 0.086 | 0.227 |
| BGE-small | 0.120 | 0.070 | 0.182 |
| gemini-embedding-2 | **0.264** | **0.180** | **0.441** |

**vs. paper** · PrivacyQA · recursive · no reranker · k=8

| Setup | Recall | Precision |
|---|---|---|
| This project (gemini-embedding-2) | 0.414 | 0.118 |
| Paper (text-embedding-3-large) | 0.424 | 0.090 |

Findings:
- Embedding model = main lever: Gemini ≈ 2× local recall@3.
- Gemini matches the paper's PrivacyQA recall@8 baseline; higher precision (smaller chunks).
- Recursive chunking helps ContractNLI; PrivacyQA within noise.
- N=194 → differences < ~0.04 are noise.

Limitations:
- Gemini = hosted API; generation not evaluated.
- Gemini embeddings tested on PrivacyQA only.
- Simplified recursive splitter, no overlap; 500-char chunks ≠ paper's chunk size.
- Brute-force retrieval; mini subset, 2 of 4 corpora evaluated.

---

## Structure

```
app.py                     Streamlit UI
config/config.yml          default index config
src/
  data_ingesting.py        corpus file listing
  chunking.py              naive + recursive chunkers
  embedding.py             MiniLM / BGE / Gemini embedders
  retrieval.py             cosine top-k
  pipeline.py              build_index() · Index (load once, query many)
  generate_answer.py       Gemini answer over chunks
  eval_metrics.py          Recall / Precision / MRR
scripts/                   build_index · evaluate · make_mini_benchmark
data/legalBench-RAG/       corpus/ · benchmarks/
indexes/<corpus>/<ts>/     saved indexes
results/<corpus>/          evaluation runs
```

---

## Reference

Pipitone & Houir Alami (2024), *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain*. [arXiv:2408.10343](https://arxiv.org/abs/2408.10343)
