import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def text_to_bert_embeddings(text, tokenizer, model):
    # Tokenize the text and convert it to input IDs, adding necessary tokens
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get the hidden states from the last layer
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden states (embedding for each token)
    last_hidden_states = outputs.last_hidden_state
    
    # Mean pooling: Calculate the mean of the hidden states to get the sentence embedding
    sentence_embedding = last_hidden_states.mean(dim=1).squeeze().numpy()
    
    return sentence_embedding

