"""
This document immplements the complete embedding pipeline 

Tokens -> Embeddings -> Search

using embedding models :

<Insert model>
"""
from dataclasses import dataclass
from src.chunking_2 import Chunk
from typing import List


@dataclass
class embedded_chunk:
    chunk: Chunk
    embedding: List[float]

def embed_text(sentences, model):
    return model.encode(sentences)

