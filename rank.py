from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def tfidf_ranking(query, documents):
    # Combine query and documents for vectorization
    corpus = [query] + documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarity between the query and each document
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_similarities.flatten()

def bm25_ranking2(query, documents):
    # Tokenize and process documents and query
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
    tokenized_query = word_tokenize(query.lower())

    # Debug: Print tokenized documents and query
    print("Tokenized Documents:", tokenized_corpus)
    print("Tokenized Query:", tokenized_query)

    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    scores = bm25.get_scores(tokenized_query)
    
    # Ensure scores are not negative and debug scores
    scores = [max(0, score) for score in scores]
    print("BM25 Scores:", scores)
    return scores

import math
from collections import Counter

def bm25_custom(query, documents, k1=1.5, b=0.75):
    # Tokenize and lowercase documents and queries
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
    tokenized_query = word_tokenize(query.lower())

    # Compute document lengths and average document length
    doc_lengths = [len(doc) for doc in tokenized_corpus]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    # Calculate document frequencies for terms in the corpus
    df = {}
    for doc in tokenized_corpus:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    # Scoring each document
    scores = []
    for doc, doc_len in zip(tokenized_corpus, doc_lengths):
        score = 0
        doc_counter = Counter(doc)
        for term in tokenized_query:
            if term in doc_counter:
                term_freq = doc_counter[term]
                doc_freq = df[term]
                idf = math.log((len(documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                term_score = idf * ((term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_length))))
                score += term_score
        scores.append(score)
    
    return scores


def bm25_ranking(query, documents):
    # Tokenize and process documents and query
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    
    # Ensure scores are not negative (this should not be necessary but added for safety)
    scores = [max(0, score) for score in scores]
    return scores




def vector_space_model(query, documents):
    # Vectorize with simple count vectors
    corpus = [query] + documents
    vectorizer = CountVectorizer(binary=True)
    count_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarity between the query and each document
    cosine_similarities = cosine_similarity(count_matrix[0:1], count_matrix[1:])
    return cosine_similarities.flatten()


queries = {
    "effects of climate change": [
        "Climate change causes rising sea levels and increased weather variability.",
        "Global warming and climate change: impacts on the environment."
    ],
    "advancements in renewable energy": [
        "Renewable energy sources like solar and wind are becoming more cost-effective.",
        "Wind turbines and solar panels as sustainable energy solutions."
    ],
    "AI in healthcare": [
        "Artificial intelligence transforms healthcare with predictive analytics.",
        "The role of AI in modern medicine and diagnosis."
    ]
}

df = pd.read_csv("updated_file.csv")

df_sorted = df.sort_values(by=['searchTerms', 'rank'], ascending=[True, False])

# Group by 'query' and collect documents into lists
query_document_mapping = df_sorted.groupby('searchTerms')['content'].apply(list).to_dict()

# Optionally, retrieve ranks for evaluation purposes
query_ranks_mapping = df_sorted.groupby('searchTerms')['rank'].apply(list).to_dict()

# print ("query_document_mapping", len(query_document_mapping.keys()))
# print ("query_ranks_mapping", query_ranks_mapping)

for row in df_sorted.iterrows():
    searchTerm = row[1][1]
    print (row[1][4])
    break

# Applying the ranking functions to each query and its associated documents
for query, docs in queries.items():
    print(f"Ranking for query: '{query}'")
    tfidf_scores = tfidf_ranking(query, docs)
    bm25_scores = bm25_custom(query, docs)
    vsm_scores = vector_space_model(query, docs)
    
    print("TF-IDF Scores:", tfidf_scores)
    print("BM25 Scores:", bm25_scores)
    print("Vector Space Model Scores:", vsm_scores)
    print("\n")
