"""
This file implements the retrieval part of the RAG system using cosine similarity and numpy arrays
"""

import numpy as np
from numpy.linalg import norm


def cosine_similarities(embedded_question, embeddings):
    return embeddings @ embedded_question / (
        norm(embeddings, axis=1) * norm(embedded_question)
    )


def retrieve_answer(embedded_question, embeddings, n_results):
    sims = cosine_similarities(embedded_question, embeddings)
    return list(np.argsort(-sims)[:n_results])


def retrieve_with_scores(embedded_question, embeddings, n_results):
    """Top-k as [(index, cosine_similarity), ...]."""
    sims = cosine_similarities(embedded_question, embeddings)
    top = np.argsort(-sims)[:n_results]
    return [(int(i), float(sims[i])) for i in top]
