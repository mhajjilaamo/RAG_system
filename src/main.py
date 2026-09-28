from src.data_ingesting import import_data_legalBench_RAG_corpus
from src.chunking import Chunk, fixed_size_chunker
from sentence_transformers import SentenceTransformer
from src.embedding import embed_text
from src.retrieval import retrieve_answer
from src.embedding import embedded_chunk



model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def print_pretty_results(question, results):
    print("Question :", question, "\n")
    #print("Listed answer in the document :", answer, "\n")
    for i in range(len(results)):
        print("Top answer n", i, " :", results[i], "\n")
        

# This pipeline loads the documents, chunks it and embeds it and sends back an array of embedded chunks
def embed_documents_pipeline(corpus):
    
    # Load corpus
    df = import_data_legalBench_RAG_corpus(corpus)
    print("loaded documents")

    # Chunk articles
    chunks = fixed_size_chunker(data=df)
    print("chunks done")

    chunks_text = [chunk.text for chunk in chunks]

    # Embed chunks
    embedded_text = embed_text(chunks_text, model)
    print("embedding chunks done")

    # Embedding dataclass
    embedded_chunks = []
    for i in range(len(chunks)):
        embedded_chunks.append(embedded_chunk(chunks[i], embedded_text[i]))

    return embedded_chunks



# This pipeline takes the user's question -> embeds it -> queries the vectore store -> question + results
def ask_vector_store_pipeline(question, embedded_chunks, n_results=3):

    # Embed question
    embedded_question = embed_text(question, model)
    print("embedding question done")

    embedded_text = [embedded_chunk.embedding for embedded_chunk in embedded_chunks] 

    # Query vectore store - retrieve answer
    result_embedded_indexes = retrieve_answer(embedded_question, embedded_text, n_results)
    print("result retrieved")

    # Transform back to text
    results = []
    for index in result_embedded_indexes:
        results.append(embedded_chunks[index].chunk)
    
    #print_pretty_results(question, results)


    return (question, results)




