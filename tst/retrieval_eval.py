"""
This file implements traditional metrics :
recall@K and Mean Reciprocal Rank (MRR)
"""
from src.main_chain import retrieval, print_pretty_results

# Returns binary 1 or 0 if answer is within k results (substring of), -1 if recall@k not computed
def recall_k(k, results, answer):
    if k<len(results):
        print("recall@k not computable : k =", k,  "is inferior to number of results returned = ", len(results))
        return -1
    i = 0
    # Check wether answer is substring of any of k results
    while i<min(k,len(results)):
        if answer in results[i]:
            return 1
        i=i+1
    return 0


def main(k):
    
    dic = retrieval(1)
    question, answer, results = dic["question"], dic["expected_answer"], dic["results"]
    
    #print_pretty_results(question, answer, results)
    score = recall_k(3, results, answer)
    print("recall@k is :", score)

    return 0

main(3)