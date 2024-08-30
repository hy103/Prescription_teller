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

title = st.text_input("Describe your problem:")

if title:
    # Display the user's input
    st.write("Your problem is:", title)
    
    # Display a message while processing
    placeholder = st.empty()
    
    # Display a message while processing
    placeholder.write("Please wait while we get the prescription for you...")
    print("You reached the please wait statement")
    
    # Get the embeddings for the input
    sentence_embeddings = get_embeddings_for_input(title)
    
    # Calculate similarity scores
    embed_df["similarity_score"] = embed_df["bert_embeddings"].apply(lambda x : cosine_similarity(x, 
                                                                       sentence_embeddings.reshape(-1,1)))
    
    # Find the drug with the highest similarity score
    recommended_drug = embed_df[embed_df["similarity_score"] == embed_df["similarity_score"].max()[0]]["Drug_Name"].iloc[0]
    
    placeholder.empty()
    
    # Display the recommended drug in a separate box with bold text
    st.markdown(
        f"""
        <div style="border: 3px solid #ddd; padding: 25px; border-radius: 5px;">
            The most relevant drug recommendation is : <strong> {recommended_drug}</strong>
        </div>
        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 20px;">
            <em>Note: The recommendation provided is for demonstration purposes only and should not be considered a formal prescription. 
            Always consult with a healthcare professional before using any medications.</em>
        </div>
        """,
        unsafe_allow_html=True
    )