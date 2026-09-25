from src.data_ingesting import load_file, load_question
from src.chunking import Chunk, fixed_size_chunker
from sentence_transformers import SentenceTransformer
from src.embedding import embed_text
from src.retrival import retrieve_answer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def print_pretty_results(question, answer, results):
    print("Question :", question, "\n")
    print("Listed answer in the document :", answer, "\n")
    for i in range(len(results)):
        print("Top answer n", i, " :", results[i], "\n")


def retrieval(question_index, n_results=3):
    
    # Load articles
    df = load_file("documents.csv")
    print("loaded documents")

    # Load question
    (question, answer) = load_question("single_passage_answer_questions.csv", question_index)
    #print("question :", question)
    #print("answer:", answer)

    # Chunk articles
    chunks = fixed_size_chunker(df=df)
    print("chunks done")

    chunks_text = [chunk.text for chunk in chunks]

    # Embed chunks
    embedded_text = embed_text(chunks_text, model)
    print("embedding chunks done")

    # Embed question
    embedded_question = embed_text(question, model)
    print("embedding question done")

    # Query vectore store - retrieve answer
    result_embedded_indexes = retrieve_answer(embedded_question, embedded_text, n_results)
    print("result retrieved")

    # Transform back to text
    results = []
    for index in result_embedded_indexes:
        results.append(chunks[index].text)
    print("results are back to text")
    print_pretty_results(question, answer, results)


    return {"question": question, "expected_answer": answer, "results": results}

#main(1)