import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_missing_rows(df):
    if all(col in df.columns for col in ["Description", "Drug_Name"]):
        # Drop rows where any of the specified columns have missing values
        return df.dropna(subset=["Description", "Drug_Name"])
    else:
        raise KeyError("One or more specified columns are not present in the DataFrame")
    

sw_nltk = stopwords.words('english')
lemmatizer = WordNetLemmatizer()



def remove_stopwords(sentence):
    words = [word for word in sentence.split() if word.lower() not in sw_nltk]
    return " ".join(words)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    #text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in sw_nltk]
    return ' '.join(words)


def cleaning_text(df, col_name):
    df["cleaned_"+col_name] = df[col_name].apply(lambda x : preprocess_text(x))

    return df


