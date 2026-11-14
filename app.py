"""
Streamlit interface: ask a question, compare retrieval (and optional answers) across embedding models and chunking strategies.

    streamlit run app.py
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pandas as pd
import streamlit as st

from src.eval_metrics import precision_k, recall_k, reciprocal_rank
from src.config import Config
from src.pipeline import Index, build_index

INDEXES_ROOT = "indexes"
CORPUS_ROOT = "data/legalBench-RAG/corpus"
GEN_TIMEOUT_S = 60   # client.interactions retries 429/503 indefinitely; cap the wait in the UI
BENCHMARKS = "data/legalBench-RAG/benchmarks"

st.set_page_config(page_title="LegalBench-RAG", layout="wide")


# ------------------------------------------------------------------ data loading
def list_indexes():
    """{corpus: {(embedding_model, chunking_strategy): index_dir}} — latest complete index per combo."""
    out = {}
    for meta_path in sorted(glob.glob(f"{INDEXES_ROOT}/*/*/index_meta.json")):   # sorted => latest wins
        d = os.path.dirname(meta_path)
        if not os.path.exists(os.path.join(d, "embeddings.npy")):
            continue
        m = json.load(open(meta_path))
        out.setdefault(m["corpus"], {})[(m["embedding_model"], m["chunking_strategy"])] = d
    return out


@st.cache_resource
def executor():
    return ThreadPoolExecutor(max_workers=4)


@st.cache_resource(show_spinner="Loading index and embedding model…")
def load_index(index_dir):
    return Index(index_dir)


@st.cache_data
def load_benchmark(corpus):
    path = f"{BENCHMARKS}/{corpus}_mini.json"
    if not os.path.exists(path):
        return []
    return json.load(open(path, encoding="utf-8"))["tests"]


@st.cache_data
def load_runs(corpus):
    rows = []
    for path in sorted(glob.glob(f"results/{corpus}/*.json")):
        d = json.load(open(path, encoding="utf-8"))
        run, agg = d.get("run", {}), d.get("aggregates", {})
        if not agg or not any(agg.values()):        # skip empty / all-zero debug runs
            continue
        rows.append({
            "date": run.get("timestamp", "")[:16].replace("T", " "),
            "embedding model": run.get("embedding model"),
            "chunking": run.get("chunk strategy"),
            "k": run.get("k"),
            "questions": run.get("num questions"),
            "recall": agg.get("mean recall"),
            "precision": agg.get("mean precision"),
            "MRR": agg.get("mean reciprocal rank"),
        })
    return pd.DataFrame(rows)


def overlaps_gold(chunk, snippets):
    end = chunk.start_index + chunk.length
    return any(chunk.source_url == s["file_path"] and chunk.start_index < s["span"][1] and end > s["span"][0]
               for s in snippets)


# ------------------------------------------------------------------ sidebar
indexes = list_indexes()

with st.sidebar:
    st.header("Settings")
    if not indexes:
        st.warning("No index yet — build one below.")
        options, chosen = {}, []
    else:
        corpus = st.selectbox("Corpus", sorted(indexes))
        available = indexes[corpus]
        models = sorted({m for m, _ in available})
        chunkings = sorted({c for _, c in available})
        sel_models = st.multiselect("Embedding models", models, default=models)
        sel_chunk = st.multiselect("Chunking", chunkings, default=chunkings,
                                   help="naive = fixed 500-char windows · recursive = split on paragraphs/"
                                        "lines/sentences, then merge up to 500 chars")
        options = {f"{m} · {c}": available[(m, c)]
                   for m in sel_models for c in sel_chunk if (m, c) in available}
        chosen = list(options)
        missing = [f"{m} · {c}" for m in sel_models for c in sel_chunk if (m, c) not in available]
        if missing:
            st.caption("Not built yet: " + ", ".join(missing))
    k = st.slider("Chunks retrieved (k)", 1, 20, 3)
    generate = st.toggle("Generate answer (Gemini)", value=False,
                         help="One Gemini API call per selected configuration.")
    st.caption("Gemini models call the hosted API and need GEMINI_API_KEY in .env.")

    with st.expander("Build an index"):
        corpora = sorted(os.listdir(CORPUS_ROOT))
        b_corpus = st.selectbox("Corpus ", corpora,
                                index=corpora.index(corpus) if indexes and corpus in corpora else 0)
        b_chunk = st.radio("Chunking ", ["naive", "recursive"], horizontal=True)
        b_model = st.radio("Embedding model ", ["minilm", "bge"], horizontal=True,
                           help="Local models only. Gemini indexes: `python -m scripts.build_index` "
                                "(slow: 60 s pause per 100 chunks for the free tier).")
        if st.button("Build"):
            with st.spinner(f"Chunking and embedding {b_corpus}…"):
                build_index(Config(corpus=b_corpus, chunking_strategy=b_chunk, embedding_model=b_model))
            st.rerun()

if not chosen:
    st.info("Select at least one embedding model and one chunking strategy that have an index.")
    st.stop()

# ------------------------------------------------------------------ main
st.title("LegalBench-RAG")
tab_ask, tab_runs = st.tabs(["Ask", "Benchmark runs"])

with tab_ask:
    tests = load_benchmark(corpus)
    mode = st.radio("Question", ["Benchmark question", "Your own question"], horizontal=True,
                    help="Benchmark questions have gold answers, so recall / precision / MRR can be computed.")
    snippets = None
    if mode == "Benchmark question" and tests:
        i = st.selectbox(f"{len(tests)} questions from {corpus}_mini.json", range(len(tests)),
                         format_func=lambda j: tests[j]["query"])
        question, snippets = tests[i]["query"], tests[i]["snippets"]
        with st.expander("Gold answer (expected snippets)"):
            for s in snippets:
                st.markdown(f"`{s['file_path']}` · chars {s['span'][0]}–{s['span'][1]}")
                st.text(s["answer"])
    else:
        question = st.text_area("Your question", placeholder="e.g. Does Viber share data with advertisers?")

    if st.button("Ask", type="primary", disabled=not (question and chosen)):
        runs = []
        for label in chosen:
            run = {"model": label}
            try:
                index = load_index(options[label])
                t0 = time.perf_counter()
                hits = index.query_with_scores(question, k)
                run["latency"] = time.perf_counter() - t0
                run["hits"] = hits
                if snippets:
                    chunks = [c for c, _ in hits]
                    run["recall"] = recall_k(chunks, snippets)
                    run["precision"] = precision_k(chunks, snippets)
                    run["rr"] = reciprocal_rank(chunks, snippets)
            except Exception as e:                       # API errors, missing key, …
                run["error"] = str(e)
            if generate and "hits" in run:
                try:
                    from src.generate_answer import generate_answer
                    t0 = time.perf_counter()
                    future = executor().submit(generate_answer, question, [c for c, _ in run["hits"]])
                    run["answer"] = future.result(timeout=GEN_TIMEOUT_S)
                    run["gen_latency"] = time.perf_counter() - t0
                except FutureTimeout:
                    run["gen_error"] = (f"no answer after {GEN_TIMEOUT_S}s — Gemini is likely rate-limited "
                                        "(quota exhausted) or overloaded")
                except Exception as e:                   # keep retrieval results if generation fails
                    run["gen_error"] = str(e)
            runs.append(run)
        st.session_state["last"] = {"question": question, "snippets": snippets, "runs": runs, "k": k}

    last = st.session_state.get("last")
    if last and last["question"] == question:
        runs, snippets = last["runs"], last["snippets"]

        # Summary statistics, one row per model
        rows = []
        for r in runs:
            row = {"configuration": r["model"]}
            if "hits" in r:
                scores = [s for _, s in r["hits"]]
                row |= {"retrieval (s)": round(r["latency"], 2),
                        "top score": round(scores[0], 3),
                        "mean score": round(sum(scores) / len(scores), 3)}
                if snippets:
                    row |= {f"recall@{last['k']}": round(r["recall"], 3),
                            f"precision@{last['k']}": round(r["precision"], 3),
                            "RR": round(r["rr"], 3),
                            "gold hits": sum(overlaps_gold(c, snippets) for c, _ in r["hits"])}
            else:
                row["error"] = r["error"][:80]
            rows.append(row)
        st.subheader("Comparison")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if snippets:
            st.caption("Recall / precision are character-level vs. the gold snippets (LegalBench-RAG metrics). "
                       "Score = cosine similarity; scores are not comparable across models.")

        # Per-model details side by side
        cols = st.columns(len(runs))
        for col, r in zip(cols, runs):
            with col:
                st.markdown(f"**{r['model']}**")
                if "error" in r:
                    st.error(r["error"])
                if "gen_error" in r:
                    st.warning(f"Answer generation failed: {r['gen_error'][:200]}")
                if "answer" in r:
                    st.info(r["answer"])
                    st.caption(f"Generated in {r['gen_latency']:.1f}s")
                for rank, (c, score) in enumerate(r.get("hits", []), start=1):
                    gold = snippets is not None and overlaps_gold(c, snippets)
                    mark = "✅ " if gold else ""
                    with st.expander(f"{mark}#{rank} · {score:.3f} · {os.path.basename(c.source_url)}"):
                        st.caption(f"chars {c.start_index}–{c.start_index + c.length}")
                        st.text(c.text)

with tab_runs:
    df = load_runs(corpus)
    if df.empty:
        st.info(f"No evaluation runs in results/{corpus}/. Run `python -m scripts.evaluate <index_dir>`.")
    else:
        st.caption(f"Saved evaluation runs on {corpus} (empty / all-zero debug runs hidden).")
        st.dataframe(df.sort_values("date", ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["recall", "precision", "MRR"]})
