from src.data_ingesting import import_data_legalBench_RAG_corpus
from src.chunking import Chunk, fixed_size_chunker, recursive_chunker
from sentence_transformers import SentenceTransformer
from src.embedding import embed_text
from src.retrieval import retrieve_answer
from src.generate_answer import generate_answer
from datetime import datetime, timezone
import json
from dataclasses import asdict
import numpy as np
import os




#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
Model_name = "BAAI/bge-small-en-v1.5"

def print_pretty_results(question, results):
    print("Question :", question, "\n")
    #print("Listed answer in the document :", answer, "\n")
    for i in range(len(results)):
        print("Top answer n", i, " :", results[i], "\n")
        

# This pipeline loads the documents, chunks it and embeds it and sends back an array of embedded chunks
def embed_documents_pipeline(corpus):
    # Generate Timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print("timestamp : ", timestamp)

    # Index directory
    index_dir = f"./index_{timestamp}/{corpus}"

    # Create index directory
    os.makedirs(index_dir, exist_ok=True)
    
    # Load corpus
    df = import_data_legalBench_RAG_corpus(corpus)
    print("loaded documents")

    # Iterate over articles
    chunks = []
    for file in df:
        path = file.path_to_corpus + file.file
        with open(path, "r", encoding="utf-8") as f:
            article = f.read()
            # Chunk article
            #chunks_article = fixed_size_chunker(article=article, source_path=file.file)
            chunks_article = recursive_chunker(article, file.file)
            chunks.extend(chunks_article)
    chunks_path = index_dir + "/chunks.json"
    # save chunks to json file
    with open(chunks_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps([asdict(c) for c in chunks], indent=4))
    
    print("chunks done")

    chunks_text = [chunk.text for chunk in chunks]

    # Embed chunks
    embedded_text = embed_text(chunks_text, model)
    embedding_path = index_dir + "/embeddings.npy"
    np.save(embedding_path, embedded_text)
    print("embedding chunks done")

 

    # write index file 
    # Timestamp
    # Corpus
    # Chunking strategy is hardcoded
    # N of chunks is len[chunks]
    # in index dire
    index_file_path = index_dir + "/index_meta.json"
    with open(index_file_path, 'w', encoding = 'utf-8') as index_f:
        index_f.write(json.dumps({"Timestamp" : timestamp,
                                  "Corpus" : corpus,
                                  "Chunking strategy" : "recursive chunker", # hardcoded for now
                                  "Number of chunks" : len(chunks),
                                  "embedding model" : Model_name}))

    return index_dir



# This pipeline takes the user's question -> embeds it -> queries the vectore store -> question + results
def ask_vector_store_pipeline(question, index_dir, n_results=3):
    # Load embeddings 
    embeddings = np.load(index_dir + "/embeddings.npy")

    # Load chunks
    with open(index_dir + "/chunks.json", "r", encoding="utf-8") as f:
        chunks = [Chunk(**d) for d in json.load(f)]

    # Embed question
    q = "Represent this sentence for searching relevant passages: " + question
    embedded_question = embed_text(q, model)
    print("embedding question done")

    # Query vectore store - retrieve answer
    result_embedded_indexes = retrieve_answer(embedded_question, embeddings, n_results)
    print("result retrieved")

    # Transform back to text
    results = []
    for index in result_embedded_indexes:
        results.append(chunks[index])
    
    print_pretty_results(question, results)


    return (question, results)

# Full end to end
# Default of the corpus = latest -> à automatiser
def main(question, index_dir ="./index_20261001_202352/privacy_qa"):

    (question_1, results) = ask_vector_store_pipeline(question, index_dir)

    answer = generate_answer(question_1, results)

    print(answer) 

    return answer



#dir_index = embed_documents_pipeline("privacy_qa")
#print(dir_index)

question = "Consider \"Viber Messenger\"'s privacy policy; does viber have any affiliation with the advertisement industry?"
index_dir = "./index_20261001_202352/privacy_qa"
main(question, index_dir)