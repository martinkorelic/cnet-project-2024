import nltk
from nltk.stem import PorterStemmer
from collections import Counter
import gensim.downloader as api
mdl = api.load("glove-wiki-gigaword-100")
from scipy.spatial.distance import cosine
import numpy as np

nltk.download('punkt')

def advancediou(array1, array2, exact_match_weight=1.0, stem_match_weight=0.5):
    stemmer = PorterStemmer()
    
    stem_to_words1 = {}
    for word in array1:
        stem = stemmer.stem(word)
        if stem not in stem_to_words1:
            stem_to_words1[stem] = set()
        stem_to_words1[stem].add(word)
    
    stem_to_words2 = {}
    for word in array2:
        stem = stemmer.stem(word)
        if stem not in stem_to_words2:
            stem_to_words2[stem] = set()
        stem_to_words2[stem].add(word)
    

    intersection_weighted_count = 0
    all_stems = set(stem_to_words1.keys()).union(stem_to_words2.keys())

    for stem in all_stems:
        words1 = stem_to_words1.get(stem, set())
        words2 = stem_to_words2.get(stem, set())
        if words1.intersection(words2):
            exact_matches = len(words1.intersection(words2))
            intersection_weighted_count += exact_matches * exact_match_weight
            
            stem_matches = min(len(words1), len(words2)) - exact_matches
            intersection_weighted_count += stem_matches * stem_match_weight
        else:
            if words1 and words2:
                stem_matches = min(len(words1), len(words2))
                intersection_weighted_count += stem_matches * stem_match_weight

    union_count = sum((Counter(array1) | Counter(array2)).values())
    
    if not union_count:
        return 0
    iou_score = intersection_weighted_count / union_count
    return iou_score

def iou(words1, words2):

    set1 = set(words1)
    set2 = set(words2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if not union:
        return 0
    iou_score = len(intersection) / len(union)
    return iou_score

def positionaliou(array1, array2):
    max_length = max(len(array1), len(array2))
    total_possible_score = len(array1) * len(array2)
    intersection_weighted_count = 0

    for index1, word1 in enumerate(array1):
        for index2, word2 in enumerate(array2):
            if word1 == word2:
                intersection_weighted_count += 1
            else:
                index_distance = abs(index1 - index2)
                penalty = index_distance * (1 / max_length)
                intersection_weighted_count += (1 - penalty)

    if total_possible_score == 0:
        return 0
    iou_score = intersection_weighted_count / total_possible_score
    return iou_score

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
    
def mean_reciprocal_rank(array1, array2):
    ranks = []
    for word1 in array1:
        for index2, word2 in enumerate(array2):
            if word1 == word2:
                ranks.append(1 / (index2 + 1))
                break
        else:
            ranks.append(0)
    if ranks:
        return sum(ranks) / len(ranks)
    else:
        return 0