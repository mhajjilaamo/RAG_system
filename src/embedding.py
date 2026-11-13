"""
Embedders. get_embedder(config) returns (embed_docs, embed_query):
- minilm / bge : local sentence-transformers models (CPU)
- gemini       : hosted gemini-embedding-2 (needs GEMINI_API_KEY in .env), 768 dims
"""
import time
from google import genai
from sentence_transformers import SentenceTransformer
from google.genai import types
from dotenv import load_dotenv


def embed_text_with_sentence_transformers(sentences, model):
    return model.encode(sentences)

# 1. Define formatting functions for Asymmetric Retrieval (Question Answering)
def prepare_document(content, title=None):
    if title is None:
        title = "none"
    return f"title: {title} | text: {content}"

def prepare_query(query):
    return f"task: question answering | query: {query}"

def load_genai_login():
    load_dotenv()
    client = genai.Client()

    return client

def embed_text_with_gemini_2(chunks_text, client):
    BATCH_SIZE = 100
    embedded_text = [] 

    for i in range(0, len(chunks_text), BATCH_SIZE):
        # 1. Get the flat list of strings for the batch
        batch = chunks_text[i : i + BATCH_SIZE]
    
        # 2. FIX: Wrap each string in its own list so the SDK treats them as separate documents
        batched_contents = [[prepare_document(text)] for text in batch]
    
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=batched_contents,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
    
        # 3. response.embeddings will now correctly contain one embedding per input string
        batch_vectors = [emb.values for emb in response.embeddings]
        embedded_text.extend(batch_vectors)

        # 4. A 60-second pause keeps you safely within the Free Tier RPM ceiling
        if i + BATCH_SIZE < len(chunks_text):
            time.sleep(60)
    return embedded_text

def embed_question_with_gemini_2(question, client):
    embedded_question = client.models.embed_content(
        model="gemini-embedding-2",
        contents=[[prepare_query(question)]],
        config=types.EmbedContentConfig(output_dimensionality=768)
    )

    # Extract the raw numeric values (.values) from each embedding object
    question_vector = embedded_question.embeddings[0].values
    return question_vector

def get_embedder(config):
    if config.embedding_model == "gemini":
        client = load_genai_login()
        return (lambda texts: embed_text_with_gemini_2(texts, client),
                lambda q:     embed_question_with_gemini_2(q, client))
    name = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "bge":    "BAAI/bge-small-en-v1.5"}[config.embedding_model]
    model = SentenceTransformer(name)
    prefix = "Represent this sentence for searching relevant passages: " if config.embedding_model == "bge" else ""
    return (lambda texts: embed_text_with_sentence_transformers(texts, model),
            lambda q:     embed_text_with_sentence_transformers(prefix + q, model))