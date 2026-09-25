"""
This file implements the retrieval part of the RAG system using cosine similarity and numpy arrays
"""

import numpy as np
from numpy.linalg import norm
import heapq

def retrieve_answer(embedded_question, embedded_text, n_results):
    results = []
    for embedding in embedded_text:
        # compute cosine similarity
        cosine = np.dot(embedding, embedded_question) / (norm(embedding) * norm(embedded_question))
        results.append(cosine)
    
    # 1. Enumerate the list to pair each item with its original index: (index, value)
    indexed_results = list(enumerate(results)) 
    # Looks like: [(0, 0.15), (1, 0.94), (2, -0.82), ...]

    # 2. Use heapq.nlargest, sorting by the value (x[1])
    top_n_tuples = heapq.nlargest(n_results, indexed_results, key=lambda x: x[1])

    return [index for index, score in top_n_tuples]



