from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import ast
import streamlit as st

### Function : TO calculate the cosine similairty between two numpy arrays
### Input : 2 numpy arrays
### output : Cosine similarity value
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product/(norm_vec1* norm_vec2)

def convert_to_list(cell_value):
    # Check if the value is not NaN
    if isinstance(cell_value, str):
        return list(map(float, cell_value.strip("[]").split()))
    return cell_value


def get_embeddings_for_input(new_sentence):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Or another pre-trained model
    new_embeddings = model.encode(new_sentence)

    return new_embeddings

def load_embeddings():

    #########Load the embeddings dataframe #############
    embed_df = pd.read_csv("sentence_embeddings.csv")
    # Apply the conversion function to the specific column
    embed_df['bert_embeddings'] = embed_df['bert_embeddings'].apply(convert_to_list)
    return embed_df


embed_df = load_embeddings()
st.markdown("# Medicine Recommender")

# Input field for user to enter their problem
title = st.text_input("Describe your problem:")
st.write("Your problem is:", title)
# Display the recommended drug
st.write("Please while we get the prescription for you")
sentence_embeddings = get_embeddings_for_input(title)
embed_df["similarity_score"] = embed_df["bert_embeddings"].apply(lambda x : cosine_similarity(x, 
                                                                       sentence_embeddings.reshape(-1,1)))

recommended_drug = embed_df[embed_df["similarity_score"] == embed_df["similarity_score"].max()[0]]["Drug_Name"].iloc[0]
st.write(f"The most relevant drug recommendation is: {recommended_drug}")