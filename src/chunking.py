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
def fixed_size_chunker(article, source_path, chunk_size = 500):
    chunks = []
    # Iterate over characters
    j=0
    while j < len(article):
        chunk = Chunk(start_index=j, source_url=source_path)
        chunk.text = article[j:j+chunk_size] 
        chunk.length = len(chunk.text)
        chunks.append(chunk)
        j = j + chunk_size
            

    return chunks 

def recursive_chunker(article, source_path, current_seperator_index, chunks_accumulator, article_start_index, max_chunk_size=500, seperators=["\n\n", "\n", ".", " ", ""]):
    chunk = Chunk(start_index= article_start_index, source_url=source_path)
    # base case : is  Text <= max_chunk_size
    if len(article) <= max_chunk_size:
        print("step 1")
        chunk.text = article
        chunk.length = len(chunk.text)
        return chunks_accumulator.append(article)
    
    # split the text by the current_seperator_index seperator
    # iterate over characters
    j=0
    article_index = article_start_index
    while j<min(len(article), max_chunk_size):
        print(article[j] == seperators[current_seperator_index])
        # if we reach seperator
        if article[j] == seperators[current_seperator_index]:
            print("step 3")
            chunk.text = article[article_start_index:j+article_start_index]
            chunk.length = len(chunk.text)
            chunks_accumulator.append(chunk)
            return recursive_chunker(article, source_path, current_seperator_index+1, chunks_accumulator,j)
        j = j + 1
    # we reached max_size before reaching a separator
    chunk.text = article[article_start_index:j+article_start_index]
    chunk.length = len(chunk.text)
    chunks_accumulator.append(chunk)
    return recursive_chunker(article, source_path, 0, chunks_accumulator, j )


def recursive_chunker(article,article_start_index, source_path, current_seperator_index, max_chunk_size=500, separators=["\n\n", "\n", ".", " ", ""]):
    if len(article) <= max_chunk_size:
        chunk = Chunk(start_index=article_start_index, source_url=source_path)
        chunk.text = article
        chunk.length = len(article)
        return [chunk]                      # a list, not a bare Chunk

    chunks = []
    offset = 0
    sep = separators[current_seperator_index]
    for split in article.split(sep):
        chunks.extend(recursive_chunker(split, article_start_index + offset, source_path,
                                    current_seperator_index + 1, max_chunk_size, separators))
        offset += len(split) + len(sep)
    return chunks

with open("data/legalBench-RAG/corpus/privacy_qa/23andMe.txt", 'r', encoding='utf-8') as file:
    article = file.read()
    chunks = recursive_chunker(article, 0 ,"privacy_qa/23andMe.txt",0 )
    #print(chunks)
    for chunk in chunks:
        print("text :", chunk.text)
        print("length :", chunk.length)
        print("start index :", chunk.start_index)
        print("\n")
