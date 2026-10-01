"""
This document immplements the embedding data class and the embedding function using the "encode" method
"""

def embed_text(sentences, model):
    return model.encode(sentences)

