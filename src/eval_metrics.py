"""
This file implements traditional metrics :
recall@K and Mean Reciprocal Rank (MRR)
"""


def recall_k(results, snippets):
    total_gold = 0
    total_covered = 0
    for snippet in snippets:
        gold = set(range(snippet['span'][0], snippet['span'][1]))
        total_gold += len(gold)

        # union of characters covered by chunks IN THIS SNIPPET'S FILE
        retrieved = set()
        for chunk in results:
            if chunk.source_url == snippet['file_path']:
                start = chunk.start_index
                end = start + chunk.length
                retrieved |= set(range(start, end))   # union — dedupes automatically

        total_covered += len(gold & retrieved)          # intersection = covered gold chars
    return total_covered / total_gold if total_gold else 0

def precision_k(results, snippets):
    covered = 0          # retrieved chars that are gold
    total_retrieved = 0  # all chars retrieved

    # total characters retrieved = union of all your chunk ranges (dedup overlaps)
    retrieved_by_file = {}
    for chunk in results:
        f = chunk.source_url   
        rng = set(range(chunk.start_index, chunk.start_index + chunk.length))
        retrieved_by_file.setdefault(f, set()).update(rng)
    total_retrieved = sum(len(s) for s in retrieved_by_file.values())

    for snippet in snippets:
        gold = set(range(snippet['span'][0], snippet['span'][1]))
        retrieved = retrieved_by_file.get(snippet['file_path'], set())
        covered += len(gold & retrieved)

    return covered / total_retrieved if total_retrieved else 0

def reciprocal_rank(results, snippets):
    for rank, chunk in enumerate(results, start=1):
        for snippet in snippets:
            if chunk.source_url == snippet['file_path']:
                start, end = chunk.start_index, chunk.start_index + chunk.length
                if start < snippet['span'][1] and end > snippet['span'][0]:  # any overlap
                    return 1 / rank
    return 0




