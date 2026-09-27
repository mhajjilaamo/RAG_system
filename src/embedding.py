"""
This document immplements the embedding data class and the embedding function using the "encode" method
"""
from dataclasses import dataclass
from src.chunking import Chunk
from typing import List


@dataclass
class embedded_chunk:
    chunk: Chunk
    embedding: List[float]

def embed_text(sentences, model):
    return model.encode(sentences)

