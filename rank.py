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
import numpy as np
import pandas as pd

def clean_df(fn):
    df = pd.read_csv(fn)
    print (df.head(10))
    filtered_df = df[df["content"].str.len() >= 4000]
    filtered_df['rank'] = filtered_df.groupby(['gl', 'searchTerms']).cumcount() + 1
    print (filtered_df.head(10))
    filtered_df = filtered_df[['gl', 'searchTerms', 'rank', 'title', 'snippet', 'link', 'content']]
    filtered_df.to_csv("reranked_data.csv")
    return filtered_df

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

def apply_ranking_algorithms(df):
    grouped = df.groupby(['gl', 'searchTerms'])
    results = []

    for name, group in grouped:
        gl, searchTerms = name
        documents_content = group['content'].tolist()
        documents_summary = group['summary'].tolist()
        indices = group.index.tolist()

        # Apply ranking algorithms to 'content'
        tfidf_scores_content = tfidf_ranking(searchTerms, documents_content)
        bm25_scores_content = bm25_custom(searchTerms, documents_content)
        vsm_scores_content = vector_space_model(searchTerms, documents_content)

        # Apply ranking algorithms to 'summary'
        tfidf_scores_summary = tfidf_ranking(searchTerms, documents_summary)
        bm25_scores_summary = bm25_custom(searchTerms, documents_summary)
        vsm_scores_summary = vector_space_model(searchTerms, documents_summary)

        # Convert scores to ranks for 'content'
        tfidf_ranks_content = np.argsort(tfidf_scores_content)[::-1] + 1
        bm25_ranks_content = np.argsort(bm25_scores_content)[::-1] + 1
        vsm_ranks_content = np.argsort(vsm_scores_content)[::-1] + 1

        # Convert scores to ranks for 'summary'
        tfidf_ranks_summary = np.argsort(tfidf_scores_summary)[::-1] + 1
        bm25_ranks_summary = np.argsort(bm25_scores_summary)[::-1] + 1
        vsm_ranks_summary = np.argsort(vsm_scores_summary)[::-1] + 1

        # Store results with indices
        for idx, tf_rank_c, bm_rank_c, vs_rank_c, tf_rank_s, bm_rank_s, vs_rank_s in zip(indices, tfidf_ranks_content, bm25_ranks_content, vsm_ranks_content, tfidf_ranks_summary, bm25_ranks_summary, vsm_ranks_summary):
            results.append((idx, tf_rank_c, bm_rank_c, vs_rank_c, tf_rank_s, bm_rank_s, vs_rank_s))

    # Convert results to DataFrame and merge
    rank_df = pd.DataFrame(results, columns=['index', 'TFIDF_Rank_Content', 'BM25_Rank_Content', 'VSM_Rank_Content', 'TFIDF_Rank_Summary', 'BM25_Rank_Summary', 'VSM_Rank_Summary'])
    df = df.merge(rank_df, left_index=True, right_on='index')

    return df



fn = "summarized_ranked.csv"
# df = clean_df(fn)
df = pd.read_csv(fn)
df = apply_ranking_algorithms(df)
df.to_csv("all_ranks_with_summary.csv")
