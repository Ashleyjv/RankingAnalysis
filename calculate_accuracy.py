
def average_precision(predicted, actual):
    hit = 0
    sum_precisions = 0
    for i, p in enumerate(predicted, start=1):
        if p in actual:
            hit += 1
            sum_precisions += hit / i
    return sum_precisions / len(actual) if actual else 0

def mean_average_precision(predictions, actuals):
    return sum(average_precision(pred, act) for pred, act in zip(predictions, actuals)) / len(actuals)

import numpy as np

def dcg(scores):
    """Calculate the Discounted Cumulative Gain for a list of scores."""
    # Convert scores to a numpy array to ensure correct operations.
    scores = np.array(scores)
    # Ensure scores are a flat array
    if scores.ndim != 1:
        raise ValueError("Scores must be a one-dimensional list of numbers.")
    return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

def ndcg(predicted, actual):
    """Calculate the Normalized Discounted Cumulative Gain for predicted and actual rankings."""
    # Create a dictionary from document IDs to relevance scores (higher is more relevant).
    relevance_scores = {doc: len(actual) - idx for idx, doc in enumerate(actual)}

    # Get the relevance scores for the predicted ordering.
    predicted_scores = [relevance_scores.get(doc, 0) for doc in predicted]

    # Calculate actual DCG using the relevance scores of predicted documents.
    actual_dcg = dcg(predicted_scores)

    # Calculate ideal DCG by sorting the actual scores in decreasing order.
    ideal_scores = sorted([relevance_scores[doc] for doc in actual], reverse=True)
    ideal_dcg = dcg(ideal_scores)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0



def precision_at_k(predicted, actual, k):
    assert k <= len(predicted)
    predicted_k = predicted[:k]
    true_hits = len(set(predicted_k) & set(actual))
    return true_hits / k


def recall_at_k(predicted, actual, k):
    assert k <= len(predicted)
    predicted_k = predicted[:k]
    true_hits = len(set(predicted_k) & set(actual))
    return true_hits / len(actual) if actual else 0



predictions = [['doc1', 'doc2', 'doc3'], ['doc5', 'doc4', 'doc6']]
actuals = [['doc3', 'doc2', 'doc1'], ['doc4', 'doc6', 'doc5']]
scores = [ndcg(pred, act) for pred, act in zip(predictions, actuals)]

print("NDCG Scores:", scores)


