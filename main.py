import pandas as pd
import numpy as np
from data_transformations import remove_missing_rows, cleaning_text
from sentence_transformers import SentenceTransformer



def main():
    ##Load data
    df = pd.read_excel("./data/archive/Medicine_description.xlsx")
    ## Removes the rows where if there is missing value in Drug names or Description.
    df = remove_missing_rows(df)
    ## Removes stop words, lower texts, lemmetize the word tokens
    new_df = cleaning_text(df, "Description")

    ## Loading Sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
    ## Applying the model 
    new_df['bert_embeddings'] = new_df['cleaned_Description'].apply(lambda x: model.encode(x))
    
    return new_df


if __name__ == '__main__':
    embeddings_df = main()
    print(embeddings_df["bert_embeddings"].shape)

    embeddings_df.to_csv("embeddings.csv", index=False)