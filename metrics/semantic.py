import gensim.downloader as api
mdl = api.load("glove-wiki-gigaword-100")
from scipy.spatial.distance import cosine
import numpy as np
def semantic(words1, words2, model=mdl):
    embeddings1 = [model[word] for word in words1 if word in model]
    embeddings2 = [model[word] for word in words2 if word in model]
    
    if not embeddings1 or not embeddings2:
        return 0.0
    
    similarities = []
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            similarity = 1 - cosine(emb1, emb2)
            similarities.append(similarity)
    
    return np.mean(similarities)