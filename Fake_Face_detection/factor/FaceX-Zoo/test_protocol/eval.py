import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate(claim_features, test_features):
    similarities = [cosine_similarity(claim, test) for claim in claim_features for test in test_features]
    return np.mean(similarities)
