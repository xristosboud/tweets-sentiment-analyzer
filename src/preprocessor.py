import re
from sklearn.base import BaseEstimator, TransformerMixin

class RegexPreprocessor:
    def __init__(self, lowercase=True, remove_user_handles=True, remove_hashtags=True, remove_numbers=True):
        self.lowercase = lowercase
        self.remove_user_handles = remove_user_handles
        self.remove_hashtags = remove_hashtags
        self.remove_numbers = remove_numbers

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        processed_texts = []
        for text in X:
            if self.lowercase:
                text.lower()
            if self.remove_user_handles:
                text = re.sub(r"@\w+(\.\w+)?\b(?!@)", "", text)
            if self.remove_hashtags:
                text = re.sub(r'#([^\s]+)', '', text)
            if self.remove_numbers:
                text = re.sub(r"\d+", "", text)
            processed_texts.append(text.strip())
            print("Processed text:", processed_texts[:5])
        return processed_texts