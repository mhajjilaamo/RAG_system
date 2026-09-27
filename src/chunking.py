""" This file implements the following chunking strategy :

< Fixed size chunking > - naive chunking 

v2 : add overlap

input : 
output : chunks

"""
from dataclasses import dataclass


@dataclass
class Chunk:
    start_index: int
    source_url: str
    length: int = 0
    text: str = ""
    
    

# Function that takes text and produces an array of 500 character chunks
def fixed_size_chunker(data, chunk_size = 500):
    chunks = []
    # Iterate over articles
    for file in data:
        path = file.path_to_corpus + file.file
        with open(path, "r", encoding="utf-8") as f:
            article = f.read()
        # Iterate over characters
        j=0
        while j < len(article):
            chunk = Chunk(start_index=j, source_url=file.file)
            chunk.text = article[j:j+chunk_size] 
            chunk.length = len(chunk.text)
            chunks.append(chunk)
            j = j + chunk_size
            

    return chunks 



