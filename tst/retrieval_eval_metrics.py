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




