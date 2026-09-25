from src.main_2 import retrieval_part1, retrival_part2
from tst.retrieval_eval_2 import recall_k
import os
import json


def main(n_results, max_test):
    embedded_chunks = retrieval_part1()
    
    average_recall = 0
    recall_sum = 0
    with open("data/legalBench-RAG/benchmarks/contractnli.json", "r", encoding="utf-8") as f:
        file = json.load(f)
        i = 0
        for querry_snippet in file["tests"]:
            question = querry_snippet["query"]
            
            # Compute results for retrival
            question, results = retrival_part2(question, embedded_chunks, n_results)

            snippets = querry_snippet["snippets"] # List of dictionnaries 

            # Compute recall@k for retrival operation
            recall = recall_k(n_results, results, snippets)

            print("question : ", question)
            print("expected answer : ")
            for snippet in snippets:
                print(snippet['answer'])

            print("Top", n_results, "results : ")
            for chunk in results:
                print(chunk.text)

            print("recall : ", recall)

            recall_sum = recall + recall_sum

            i = i + 1
            if i==max_test:
                break

    average_recall = recall_sum / max_test
    return average_recall

main(3,2)