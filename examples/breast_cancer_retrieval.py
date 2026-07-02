"""Retrieval with tabular foundation model embeddings, sentence-transformers style.

Embeds the breast cancer dataset with TabICLv2 and retrieves the most
similar corpus rows for a set of query rows.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tfm_embeddings import TabularEmbedder

X, y = load_breast_cancer(return_X_y=True)
X_corpus, X_query, y_corpus, y_query = train_test_split(X, y, test_size=100, random_state=0)

# 1. Load a tabular foundation model
model = TabularEmbedder("tabicl")

# 2. Set the context table -- all embeddings are conditioned on it
model.fit(X_corpus, y_corpus)

# 3. Calculate embeddings by calling model.encode()
embeddings = model.encode(X_corpus)
print(embeddings.shape)
# (469, 512)

# 4. Calculate the embedding similarities
query_embeddings = model.encode(X_query[:3])
similarities = model.similarity(query_embeddings, query_embeddings)
print(similarities.round(4))

# 5. Retrieve the most similar corpus rows for each query
indices, scores = model.search(X_query[:3], top_k=5)
for q, (idx, score) in enumerate(zip(indices, scores)):
    neighbor_labels = y_corpus[idx].tolist()
    print(
        f"query {q} (label={y_query[q]}): "
        f"top-5 neighbor labels={neighbor_labels}, scores={score.round(3).tolist()}"
    )
