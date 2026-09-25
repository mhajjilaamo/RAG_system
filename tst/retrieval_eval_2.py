"""
This file implements traditional metrics :
recall@K and Mean Reciprocal Rank (MRR)
"""


def recall_k(k, results, snippets):
    if len(results)<k:
        print("recall@k not computable : k =", k, "is superior to number of results returned = ", len(results))
        return -1
    
    # Compare spans and source files of results and snippets
    # Snippers is an array of multiple passages
    # We need to check wether all of the snippets are included in at least on of the results
    for snippet in snippets:
        covered = False
        for chunk in results:
            if snippet['file_path'] == chunk.source_url:
                chunk_end = chunk.start_index + chunk.length
                if chunk.start_index <= snippet['span'][0] and chunk_end >= snippet['span'][1]:
                    covered = True
                    break
        if not covered:
            return 0   # a snippet nobody covered → whole question misses
    return 1           # every snippet was covered
    




