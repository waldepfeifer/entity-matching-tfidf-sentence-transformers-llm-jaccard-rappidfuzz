import os
import re
import pandas as pd
import spacy
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_from_folder(folder_path):
    """Load CSV files for Retailer A and B datasets from the specified folder."""
    retailerA_path = os.path.join(folder_path, 'DDB_retailerA.csv')
    retailerB_path = os.path.join(folder_path, 'DDB_retailerB.csv')

    if not os.path.exists(retailerA_path) or not os.path.exists(retailerB_path):
        raise FileNotFoundError("Both DDB_retailerA.csv and DDB_retailerB.csv must be present in the folder.")

    retailerA_df = pd.read_csv(retailerA_path)
    retailerB_df = pd.read_csv(retailerB_path)

    return retailerA_df, retailerB_df

def clean_text_spacy(text):
    """Clean and preprocess text using regular expressions."""
    if pd.isnull(text):  # Handle Null values
        return ""
    text = text.lower()  # Lowercase text
    text = re.sub(r'[\"\'\-]', '', text)  # Remove quotes and dashes
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Replace non-alphanumeric characters with space
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces to a single space
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove standalone single letters
    return text

def tokenize_and_lemmatize(text, nlp):
    """Tokenize and lemmatize text using spaCy."""
    text = clean_text_spacy(text)  # Clean text
    doc = nlp(text)  # Tokenize
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]  # Lemmatize, keep alphabetic tokens, filter stopwords, and remove short tokens (< 3 chars)
    return ' '.join(tokens)  # Join tokens into clean string

def clean_data(retailerA_df, retailerB_df):
    """Clean the columns of Retailer A and Retailer B datasets."""
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Clean Retailer A columns
    retailerA_df['name'] = retailerA_df['name'].apply(lambda x: clean_text_spacy(x))
    #retailerA_df['description'] = retailerA_df['description'].apply(lambda x: tokenize_and_lemmatize(x, nlp))
    #retailerA_df['manufacturer'] = retailerA_df['manufacturer'].fillna('')  # Handle null values

    # Clean Retailer B columns
    retailerB_df['title'] = retailerB_df['title'].apply(lambda x: clean_text_spacy(x))
    #retailerB_df['description'] = retailerB_df['description'].apply(lambda x: tokenize_and_lemmatize(x, nlp))
    #retailerB_df['manufacturer'] = retailerB_df['manufacturer'].fillna('')  # Handle null values

    return retailerA_df, retailerB_df

def compute_matches(retailerA_df, retailerB_df, output_file):
    """Compute TF-IDF + Cosine Similarity to generate matching pairs."""

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the combined text from both datasets
    tfidf_retailerA = tfidf_vectorizer.fit_transform(retailerA_df['name'])
    tfidf_retailerB = tfidf_vectorizer.transform(retailerB_df['title'])

    # Compute cosine similarity between all pairs
    cosine_sim_matrix = cosine_similarity(tfidf_retailerA, tfidf_retailerB)

    # Define a threshold for matching
    similarity_threshold = 0.705

    # Generate matching pairs
    matches = []
    for i in range(cosine_sim_matrix.shape[0]):
        for j in range(cosine_sim_matrix.shape[1]):
            match_flag = 1 if cosine_sim_matrix[i, j] >= similarity_threshold else 0
            matches.append({
                'retailerA_id': retailerA_df.iloc[i]['id'],
                'retailerB_id': retailerB_df.iloc[j]['id'],
                'match': match_flag,
            })

    # Convert matches to a DataFrame and save to CSV
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(output_file, index=False)

    print(f"TF-IDF + Cosine Similarity matching completed and saved to {output_file}.")

def main(input_folder, output_file):
    print("Loading datasets from folder:", input_folder)
    retailerA_df, retailerB_df = load_data_from_folder(input_folder)

    print("Cleaning data...")
    retailerA_df, retailerB_df = clean_data(retailerA_df, retailerB_df)

    print("Computing matches...")
    compute_matches(retailerA_df, retailerB_df, output_file)

    print("Process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Matching Data Preprocessing")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the folder containing the datasets.")
    parser.add_argument("-o", "--output_file", required=True, help="Path to save the match report.")

    args = parser.parse_args()

    main(args.input_folder, args.output_file)
