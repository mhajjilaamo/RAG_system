"""

This file provides an interface to read the files in a corpus for the LegalBench-RAG dataset 

"""

import os
import json
from dataclasses import dataclass


@dataclass
class data:
    path_to_corpus: str #path
    file: str   #path
    corpus: str 




def import_data_legalBench_RAG_corpus(corpus):
    text_data = []
    corpus_dir = "data/legalBench-RAG/corpus/" + corpus   # the folder to READ from
    for file in os.listdir(corpus_dir):
        text_data.append(data(
            path_to_corpus="data/legalBench-RAG/corpus/",  # for building read path
            file=corpus + "/" + file,                      # matches benchmark file_path
            corpus=corpus,
        ))
    return text_data

#def load_question(index):
#    with open("data/legalBench-RAG/benchmarks/contractnli.json", "r", encoding="utf-8") as f:
#        file = json.load(f)
#        question = file["tests"][index]["query"]
#        answer = file["tests"][index]["snippets"][0]['answer']
#        return(question, answer)

#print(import_data_legalBench_RAG_corpus("contractnli"))
#print(load_question(1))
