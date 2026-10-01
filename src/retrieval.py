"""
This file implements the retrieval part of the RAG system using cosine similarity and numpy arrays
"""

import numpy as np
from numpy.linalg import norm
import heapq

def retrieve_answer(embedded_question, embeddings, n_results):
    sims = embeddings @ embedded_question / (
        norm(embeddings, axis=1) * norm(embedded_question)
    )
    return list(np.argsort(-sims)[:n_results])


