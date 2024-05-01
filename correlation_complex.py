import pandas as pd
from scipy.stats import spearmanr, kendalltau

# Load your DataFrame from a CSV file
df = pd.read_csv("all_ranks_with_summary.csv")

def calculate_context_specific_correlations(df):
    grouped = df.groupby(['gl', 'searchTerms'])
    results = []
    
    # Initialize lists to store overall correlation values for averaging later
    spearman_content = []
    kendall_content = []
    spearman_summary = []
    kendall_summary = []

    for name, group in grouped:
        gl, searchTerms = name
        # Ensure there are no missing values in the group for the columns used
        group = group.dropna(subset=['rank', 'TFIDF_Rank_Content', 'BM25_Rank_Content', 'VSM_Rank_Content', 
                                     'TFIDF_Rank_Summary', 'BM25_Rank_Summary', 'VSM_Rank_Summary'])

        # Calculate Spearman's and Kendall's correlations if enough data
        if len(group) > 1:  # Ensuring there are at least two data points to calculate correlation
            # Correlations for content
            spearman_tfidf_content = spearmanr(group['rank'], group['TFIDF_Rank_Content']).correlation
            kendall_tfidf_content = kendalltau(group['rank'], group['TFIDF_Rank_Content']).correlation

            spearman_bm25_content = spearmanr(group['rank'], group['BM25_Rank_Content']).correlation
            kendall_bm25_content = kendalltau(group['rank'], group['BM25_Rank_Content']).correlation

            spearman_vsm_content = spearmanr(group['rank'], group['VSM_Rank_Content']).correlation
            kendall_vsm_content = kendalltau(group['rank'], group['VSM_Rank_Content']).correlation

            # Correlations for summary
            spearman_tfidf_summary = spearmanr(group['rank'], group['TFIDF_Rank_Summary']).correlation
            kendall_tfidf_summary = kendalltau(group['rank'], group['TFIDF_Rank_Summary']).correlation

            spearman_bm25_summary = spearmanr(group['rank'], group['BM25_Rank_Summary']).correlation
            kendall_bm25_summary = kendalltau(group['rank'], group['BM25_Rank_Summary']).correlation

            spearman_vsm_summary = spearmanr(group['rank'], group['VSM_Rank_Summary']).correlation
            kendall_vsm_summary = kendalltau(group['rank'], group['VSM_Rank_Summary']).correlation

            results.append({
                'gl': gl,
                'searchTerms': searchTerms,
                'Spearman_TFIDF_Content': spearman_tfidf_content,
                'Kendall_TFIDF_Content': kendall_tfidf_content,
                'Spearman_BM25_Content': spearman_bm25_content,
                'Kendall_BM25_Content': kendall_bm25_content,
                'Spearman_VSM_Content': spearman_vsm_content,
                'Kendall_VSM_Content': kendall_vsm_content,
                'Spearman_TFIDF_Summary': spearman_tfidf_summary,
                'Kendall_TFIDF_Summary': kendall_tfidf_summary,
                'Spearman_BM25_Summary': spearman_bm25_summary,
                'Kendall_BM25_Summary': kendall_bm25_summary,
                'Spearman_VSM_Summary': spearman_vsm_summary,
                'Kendall_VSM_Summary': kendall_vsm_summary
            })

            # Collect data for average calculation
            spearman_content.append(spearman_tfidf_content)
            kendall_content.append(kendall_tfidf_content)
            spearman_summary.append(spearman_tfidf_summary)
            kendall_summary.append(kendall_tfidf_summary)

    # Calculate average correlations
    avg_spearman_content = sum(spearman_content) / len(spearman_content) if spearman_content else 0
    avg_kendall_content = sum(kendall_content) / len(kendall_content) if kendall_content else 0
    avg_spearman_summary = sum(spearman_summary) / len(spearman_summary) if spearman_summary else 0
    avg_kendall_summary = sum(kendall_summary) / len(kendall_summary) if kendall_summary else 0

        # Print average correlations for a quick overview
    print("Average Spearman's Rank Correlation for Content: {:.4f}".format(avg_spearman_content))
    print("Average Spearman's Rank Correlation for Summary: {:.4f}".format(avg_spearman_summary))
    print("Average Kendall's Tau Correlation for Content: {:.4f}".format(avg_kendall_content))
    print("Average Kendall's Tau Correlation for Summary: {:.4f}".format(avg_kendall_summary))

    # Compare average correlations and append to results for output
    results.append({
        'Comparison_Type': 'Average',
        'Spearman_Content': avg_spearman_content,
        'Spearman_Summary': avg_spearman_summary,
        'Kendall_Content': avg_kendall_content,
        'Kendall_Summary': avg_kendall_summary
    })

    # Convert results to DataFrame for easier analysis and export
    result_df = pd.DataFrame(results)
    return result_df







# Call the function and print the results
context_correlations = calculate_context_specific_correlations(df)
print(context_correlations)

# Optionally, save to CSV
context_correlations.to_csv('context_specific_correlations.csv')
