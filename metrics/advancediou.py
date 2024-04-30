import nltk
from nltk.stem import PorterStemmer
from collections import Counter

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
