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


def recursive_split(text, start_index, separators, max_chunk_size):
    """Flat list of (piece_text, abs_start_index), each <= max_chunk_size where possible.
    start_index is where `text` begins in the original article."""
    if len(text) <= max_chunk_size:
        return [(text, start_index)]          # already fits — atomic piece

    if not separators:                        # ran out of separators: hard-cut by size
        return [(text[i:i+max_chunk_size], start_index + i)
                for i in range(0, len(text), max_chunk_size)]

    sep = separators[0]
    parts = list(text) if sep == "" else text.split(sep)   # "" => split into characters

    pieces = []
    offset = 0
    for part in parts:
        pieces.extend(recursive_split(part, start_index + offset, separators[1:], max_chunk_size))
        offset += len(part) + len(sep)        # advance past the piece AND its separator
    return pieces

def merge_pieces(pieces, article, source_path, max_chunk_size):
    """Greedily pack adjacent pieces into chunks <= max_chunk_size."""
    def make_chunk(start, end):
        c = Chunk(start_index=start, source_url=source_path)
        c.text = article[start:end]           # slice from original => text matches start_index
        c.length = len(c.text)
        return c

    if not pieces:
        return []

    chunks = []
    buffer_start = pieces[0][1]
    buffer_end   = pieces[0][1] + len(pieces[0][0])

    for text, start in pieces[1:]:
        piece_end = start + len(text)
        if piece_end - buffer_start > max_chunk_size:   # adding this piece would overflow
            chunks.append(make_chunk(buffer_start, buffer_end))   # flush
            buffer_start, buffer_end = start, piece_end           # start fresh
        else:
            buffer_end = piece_end                                # extend buffer
    chunks.append(make_chunk(buffer_start, buffer_end))           # final flush
    return chunks


def recursive_chunker(article, source_path, max_chunk_size=500,
                      separators=["\n\n", "\n", ".", " ", ""]):
    pieces = recursive_split(article, 0, separators, max_chunk_size)
    return merge_pieces(pieces, article, source_path, max_chunk_size)

# with open("data/legalBench-RAG/corpus/privacy_qa/23andMe.txt", 'r', encoding='utf-8') as file:
#     article = file.read()
#     chunks = recursive_chunker(article, "privacy_qa/23andMe.txt")
#     #print(chunks)
#     for chunk in chunks:
#         print("text :", chunk.text)
#         print("length :", chunk.length)
#         print("start index :", chunk.start_index)
#         print("\n")
