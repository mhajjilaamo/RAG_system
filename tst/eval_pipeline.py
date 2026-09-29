from src.main import embed_documents_pipeline, ask_vector_store_pipeline
from tst.retrieval_eval_metrics import recall_k, precision_k, reciprocal_rank
import os
import json
from datetime import datetime, timezone
from dataclasses import asdict



def main(corpus, n_results, max_test):
    # Create results file
    timestamp = datetime.now(timezone.utc).isoformat()
    print("timestamp : ", timestamp)


    run = {'timestamp' : timestamp, 
           'dataset' : "LEGAL BENCH RAG",
           'corpus' : corpus,
           'num questions' : max_test,
           'chunk strategy' : "naive chunk",
           'chunk size' : 500,
           'chunk overlap' : 0,
           'embedding model' : "sentence-transformers/all-MiniLM-L6-v2",
           'k': n_results}
     
    eval_results = {"run" : run,
               "aggregates" : {},
               "per_question" : []}

    # Embed database
    embedded_chunks = embed_documents_pipeline(corpus)
    
    average_recall = 0
    average_precision = 0
    average_reciprocal_rank = 0

    recall_sum = 0
    precision_sum = 0
    reciprocal_rank_sum = 0
    corpus_benchmark_path = "data/legalBench-RAG/benchmarks/" + corpus + "_mini.json"
    with open(corpus_benchmark_path, "r", encoding="utf-8") as f:
        file = json.load(f)
        i = 0
        for querry_snippet in file["tests"]:
            question = querry_snippet["query"]
            
            # Compute results for retrival (results is a list of chunks)
            question, results = ask_vector_store_pipeline(question, embedded_chunks, n_results)

            snippets = querry_snippet["snippets"] # List of dictionnaries 

            # Compute recall@k for retrival operation
            recall = recall_k(results, snippets)
            precision = precision_k(results, snippets)
            reciprocal_r = reciprocal_rank(results, snippets)


            print("question : ", question)
            print("expected answer : ")
            for snippet in snippets:
                print(snippet['answer'])

            print("Top", n_results, "results : ")
            for chunk in results:
                print(chunk.text)

            print("recall : ", recall)
            print("precision : ", precision)
            print("reciprocal rank : ", reciprocal_r)
            print("\n")

            recall_sum = recall + recall_sum
            precision_sum = precision + precision_sum
            reciprocal_rank_sum = reciprocal_r + reciprocal_rank_sum

            # json dump per question results to file
            per_question = {"question" : question,
                            "gold snippets" : snippets,
                            "retrieved results" : [asdict(result) for result in results],
                            "recall" : recall,
                            "precision" : precision,
                            "reciprocal rank" : reciprocal_r}
            
            eval_results['per_question'].append(per_question)
            


            i = i + 1
            if i==max_test:
                break

    average_recall = recall_sum / max_test
    print("average recall :", average_recall)

    average_precision = precision_sum / max_test
    print("average precision :", average_precision)

    average_reciprocal_rank = reciprocal_rank_sum / max_test
    print("average reciprocal rank :", average_reciprocal_rank)

    # json dump agregates to file
    aggregates = {"mean recall" : average_recall,
                  "mean precision" : average_precision,
                  "mean reciprocal rank" : average_reciprocal_rank}
    
    eval_results['aggregates'] = aggregates
    
    # json dump data to file
    file_name = "run_" + corpus + "_" + str(timestamp) + ".json"
    path = "./results/" + file_name 
    with open(path, "w") as file:
        file.write(json.dumps(eval_results, indent=4))
    


    return average_recall

main("privacy_qa",3,3)
print("privacy qa done")

#main("contractnli",3, 194)
#print("contract_nli done")