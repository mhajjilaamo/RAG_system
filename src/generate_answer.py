from google import genai
from dotenv import load_dotenv
from src.chunking import Chunk



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

    return 0

#question = "Consider \"Viber Messenger\"'s privacy policy; does viber have any affiliation with the advertisement industry?"
#results = [Chunk(21977, "privacy_qa/Viber Messenger.txt", 357, "We encourage you to contact us to update or correct your information if it changes or if the personal information we hold about you is inaccurate.\nViber Media S.a.r.l is the controller of data we collect and process according to this privacy policy.\nWe are committed to working with you to obtain a fair resolution of any complaint or concern about privacy."),
#           Chunk(13453,"privacy_qa/Viber Messenger.txt", 14416,"(b) App Providers and Other Third-Parties:We may disclose your information to service providers and other third-parties under contract who help with providing you and others our Services on our behalf or other services provided by third-parties via our Services (such as, but not limited to, fraud and spam investigations, payment processing, site analytics and operations, providing special partnership features in our service either on an aggregate non identifiable basis, or using a unique identifier which is not attributable to you).\nThey are required to secure the data they receive.\n(c) Advertising partners: to enable the limited advertisements on our service, we may share a unique advertising identifier that is not attributable to you, with our third party advertising partners, and advertising service providers, along with certain technical data about you (your language preference, country, city, and device data), based on our legitimate interest.")]

#generate_answer(question, results)