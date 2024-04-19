from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample documents
documents = [
    "the sky is blue",
    "sky is clear and blue",
    "the sun is bright",
    "sunshine and clear skies",
    "the sun in the blue sky"
]

# Query
query = ["the blue sky"]

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_documents = vectorizer.fit_transform(documents)

# Transform the query using the same vectorizer
tfidf_query = vectorizer.transform(query)

# Compute cosine similarity between the query and documents
cosine_similarities = cosine_similarity(tfidf_query, tfidf_documents).flatten()

# Rank documents based on cosine similarity
ranked_documents = np.argsort(-cosine_similarities)  # argsort in descending order

# Display the ranked documents along with their scores
print("Ranked Documents based on query relevance:")
for index in ranked_documents:
    print(f"Document: '{documents[index]}', Score: {cosine_similarities[index]:.4f}")