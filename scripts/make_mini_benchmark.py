import json, random

def make_mini(corpus, n_per_corpus=194, seed=42):
    """Sample a balanced subset from a full LegalBench-RAG benchmark file.
    Reproducible via fixed seed; writes {corpus}_mini.json."""
    path = f"data/legalBench-RAG/benchmarks/{corpus}.json"
    with open(path, "r", encoding="utf-8") as f:
        full = json.load(f)

    tests = full["tests"]
    rng = random.Random(seed)                      # local RNG, doesn't touch global state
    k = min(n_per_corpus, len(tests))
    sampled = rng.sample(tests, k)                 # sample WITHOUT replacement

    mini = dict(full)                              # keep any top-level metadata the file has
    mini["tests"] = sampled

    out = f"data/legalBench-RAG/benchmarks/{corpus}_mini.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mini, f, indent=2, ensure_ascii=False)
    print(f"{corpus}: sampled {k} of {len(tests)} → {out}")

for c in ["contractnli", "cuad", "maud", "privacy_qa"]:   # verify exact names first
    make_mini(c)