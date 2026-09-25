""" This file implements the following chunking strategy :

< Fixed size chunking >

v2 : add overlap

input : panda dataframe csv 
output : chunks

"""

from dataclasses import dataclass
from src.data_ingesting import load_file



@dataclass
class Chunk:
    start_index: int
    source_url_index: int
    length: int = 0
    text: str = ""
    
    

# Function that takes text and produces an array of 500 character chunks
def fixed_size_chunker(df, chunk_size = 500):
    chunks = []
    # Iterate over articles
    for i in range(df.shape[0]):
        article = df['text'][i]
        # Iterate over characters
        j=0
        while j < len(article):
            chunk = Chunk(start_index=j, source_url_index=i)
            chunk.text = article[j:j+chunk_size] 
            chunk.length = len(chunk.text)
            chunks.append(chunk)
            j = j + chunk_size
            

    return chunks 



def main():
    df = load_file("documents.csv")
    chunks = fixed_size_chunker(df=df)
    print(chunks[190])
    print(chunks[190].length)
    print(chunks[190].source_url_index)
    return chunks

if __name__ == "__main__":
    main()