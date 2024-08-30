Medicine Recommender
Welcome to the Medicine Recommender project! This project aims to provide personalized medicine recommendations based on user input symptoms. By leveraging state-of-the-art natural language processing (NLP) techniques, this API suggests appropriate medications based on the given medical condition.

Overview
In today's world, maintaining privacy and security of personal medical data is crucial. This project seeks to address the need for a personal medicine recommender that keeps medical history secure and provides relevant prescriptions based on user symptoms.

Dataset
The dataset used in this project is the "Medicine Descriptions" dataset from Kaggle. It includes the following columns:

Drug_Name: The name of the medication prescribed.
Reason: The main reason for the medication.
Description: A detailed description of the prescription.
Data Cleaning
During the initial exploration of the dataset, I noticed that there were a few missing values. Since these missing values were less than 1% of the total data, I have removed the rows with missing information. The cleaned dataset includes information on various medical issues, with notable conditions being Supplement, Pain, Infection, Hypertension, Diabetes, Fungal Diseases, Allergies, and Depression.

Data Analysis
Distribution of Medical Issues
We analyzed the distribution of medical issues and found that conditions like Depression are increasingly prevalent. This was visualized using a pie chart.

Most Frequent Words
We examined the most frequent words in the prescriptions. Terms like "treatment," "blood," "blood pressure," and "diabetes" were commonly mentioned. This was visualized using a bar plot.

Data Preprocessing
The text data underwent several preprocessing steps:

Removal of stop words
Tokenization
Lemmatization
Text Embeddings

SBERT was chosen for its ability to generate and compare sentence embeddings effectively. Built on top of BERT, SBERT is optimized for semantic similarity and sentence retrieval, providing fixed-size representations for sentences.

Text Similarity
We employed cosine similarity to compare vector embeddings. Other methods for text similarity include Mean Reciprocal Rank, Spearman Correlation, and Pearson Correlation, but cosine similarity was used for its effectiveness in our use case.

Making Predictions
When a user inputs a medical issue, the sentence is converted into vector embeddings using SBERT. We calculate cosine similarity scores between the user's input and existing descriptions from the dataset. The description with the highest cosine similarity score, along with its corresponding drug name, is selected as the recommended prescription.

Future Plans
Dataset Expansion: Increase the size of the dataset to improve the model's accuracy.
User Experience: Enhance user experience by validating their issue with the "Reason" from the dataset.
Usage
To use the medicine recommender API, follow these steps:

Clone the Repository:

bash
Copy code
git clone https://github.com/hy103/Prescription_teller.git

Install Dependencies: Navigate to the project directory and install the required Python packages:

bash
Copy code
cd medicine-recommender
pip install -r requirements.txt
Run the API:

bash
Copy code
streamlit run Medicine_recommender_api.py
Access the API: Use the provided endpoints to interact with the recommender system.

Contributing
Feel free to contribute to this project by opening issues, submitting pull requests, or providing feedback. Your contributions are welcome!

Acknowledgments
Special thanks to Kaggle for providing the dataset.

Contact
For any questions or suggestions, please reach out to 24y.harsha@gmail.com





