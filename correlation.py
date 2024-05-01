import pandas as pd
from scipy.stats import spearmanr, kendalltau

# Assume df is your DataFrame after the ranking algorithms have been applied and merged back
df = pd.read_csv("all_ranks.csv")
def calculate_correlations(df):
    # Spearman's Rank Correlation
    spearman_tfidf = spearmanr(df['rank'], df['TFIDF_Rank']).correlation
    spearman_bm25 = spearmanr(df['rank'], df['BM25_Rank']).correlation
    spearman_vsm = spearmanr(df['rank'], df['VSM_Rank']).correlation

    # Kendall's Tau
    kendall_tfidf = kendalltau(df['rank'], df['TFIDF_Rank']).correlation
    kendall_bm25 = kendalltau(df['rank'], df['BM25_Rank']).correlation
    kendall_vsm = kendalltau(df['rank'], df['VSM_Rank']).correlation

    # Print or return the results
    print("Spearman's Rank Correlation:")
    print(f"TF-IDF: {spearman_tfidf}, BM25: {spearman_bm25}, VSM: {spearman_vsm}")

    print("Kendall's Tau Correlation:")
    print(f"TF-IDF: {kendall_tfidf}, BM25: {kendall_bm25}, VSM: {kendall_vsm}")

    return {
        "spearman": {"TFIDF": spearman_tfidf, "BM25": spearman_bm25, "VSM": spearman_vsm},
        "kendall": {"TFIDF": kendall_tfidf, "BM25": kendall_bm25, "VSM": kendall_vsm}
    }

# Sample call to the function
correlations = calculate_correlations(df)
