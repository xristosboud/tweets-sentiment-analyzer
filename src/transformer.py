import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        embeddings = self.model.encode(X, convert_to_numpy=True)
        print(f"Shape of embeddings: {embeddings.shape}")
        return embeddings