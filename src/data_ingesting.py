import kagglehub
import pandas as pd


def call_data():
    # Download latest version
    download_path = kagglehub.dataset_download("samuelmatsuoharris/single-topic-rag-evaluation-dataset")
    return download_path
    
# Converts to panda dataframes
def load_file(document):
    path = call_data()
    df = pd.read_csv(path + "/" +document)
    return df

def load_question(document, question_index):
    file = load_file(document)
    return ((file['question'][question_index], file['answer'][question_index]))

def main():
    df = load_file("documents.csv")
    print(df.head())
    print(df['text'][2])
    return 0

if __name__ == "__main__":
    main()


