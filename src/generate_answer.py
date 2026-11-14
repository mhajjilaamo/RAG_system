"""Grounded answer generation over retrieved chunks (Gemini). Demonstration layer, not evaluated."""
from google import genai
from dotenv import load_dotenv


def pretty_print(question, passed_answer, generated_answer):
    print("Question :", question)
    print("Generated_answer :", generated_answer)
    
    for index, answer in enumerate(passed_answer):
        print("Passed_answer n :",index, answer)

    return 0


def generate_answer(question, results):
    document_extracts = [result.text for result in results]
    prompt=f"Turn the following document extracts {document_extracts} into an answer to the question when relevant : question: {question}"


    # Load environment variables from the .env file
    load_dotenv()

    # The client will now successfully detect GEMINI_API_KEY
    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3.7-flash",
        input=prompt
    )

    pretty_print(question, document_extracts, interaction.output_text)
    return interaction.output_text
