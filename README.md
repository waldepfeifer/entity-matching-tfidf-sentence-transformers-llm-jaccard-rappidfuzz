<img width="1324" alt="image" src="https://github.com/user-attachments/assets/1b83df1e-e02f-4e03-9175-56dfa21bfb8b" />


## Project Overview

This project performs entity matching between two product catalogs from different retailers using both traditional NLP and transformer-based techniques.  
The goal is to identify semantically similar or identical products across datasets, despite differences in naming conventions or textual formatting.  
The workflow combines TF-IDF vectorization and BERT-based embeddings with cosine similarity scoring to generate candidate matches.

## Objectives

- Align product records from Retailer A and Retailer B  
- Engineer meaningful text representations using TF-IDF and Sentence-BERT  
- Compute cosine similarity scores between encoded product fields  
- Apply thresholding to identify probable matches  
- Evaluate similarity-based results for precision and coverage  

## Tools and Libraries

- Python 3.x  
- pandas  
- numpy  
- scikit-learn (TfidfVectorizer, cosine_similarity)  
- sentence-transformers (BERT embeddings)  
- tqdm (for progress display)  
- seaborn / matplotlib (for optional visualization)

## Project Structure

entity-matching-tfidf-bert-retail-products/  
├── product_entity_matching_with_tfidf_and_bert.ipynb   – Main notebook with full matching logic  
├── DDB_retailerA.csv                                   – First product catalog  
├── DDB_retailerB.csv                                   – Second product catalog  
├── README.md                                           – Project documentation  

## Methodology

1. **TF-IDF Vectorization**
   - Text fields: product name, brand, and category  
   - Vectorized using `TfidfVectorizer` with n-gram and lowercase options  
   - Cosine similarity calculated between TF-IDF representations  

2. **Sentence-BERT Embeddings**
   - Model: `all-MiniLM-L6-v2` or equivalent  
   - Encodes sentences into dense 384-dim vectors  
   - Cosine similarity between embeddings provides deep semantic matching  

3. **Threshold-Based Matching**
   - Matches are defined when similarity score exceeds a defined threshold (e.g., 0.85)  
   - Output includes sorted match suggestions for human review  

4. **Evaluation**
   - Rank-ordered matches based on similarity  
   - Manual validation or labeling of top matches  
   - Can be extended with classification pipeline or clustering

## How to Run

1. Ensure Python 3.x is installed  
2. Install dependencies using pip:  
   pip install pandas numpy scikit-learn sentence-transformers tqdm

3. Place both retailer CSV files in the same directory  
4. Open and run: `product_entity_matching_with_tfidf_and_bert.ipynb`

## Requirements

Install all required packages using:  
pip install pandas numpy scikit-learn sentence-transformers tqdm

## License

This project is licensed under the MIT License. See the LICENSE file for details.
