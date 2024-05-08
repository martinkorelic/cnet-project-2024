import nltk, json
import pandas as pd
from nltk.stem import PorterStemmer
from collections import Counter
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

def cosine_distance(words1, words2, query, model):

    embeddings1 = [model[word] for word in words1 if word in model]
    embeddings2 = [model[word] for word in words2 if word in model]
    
    if not embeddings1 or not embeddings2:
        return 0.0
    
    root_embedding = model[query]
    
    similarities = []
    emb1_to_root = []
    emb2_to_root = []
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            similarity = 1 - cosine(emb1, emb2)
            similarities.append(similarity)

            # Compute also distance to root query word
            emb1_to_root.append(1 - cosine(root_embedding, emb1))
            emb2_to_root.append(1 - cosine(root_embedding, emb2))

    return np.mean(similarities), np.mean(emb1_to_root), np.mean(emb2_to_root)

def wm_distance(words1, words2, model):
    wm = model.wmdistance(words1, words2)
    if wm >= 1:
        return 0.0
    return 1 - wm

def accuracy(y_true, y_pred):
    y_pred = set(y_pred)
    y_true = set(y_true)
    return len(y_true.intersection(y_pred)) / len(y_true)

def run_evaluation(query, ref_models=None, result_path='results', words_path='wordsdata', algos = ['rw', 'node2vec', 'struc2vec', 'deepwalk']):

    df = {
        # Name of comparison
        'name': [],
        # Metrics
        'IoU': [],
        'advIoU': [],
        'posIoU': [],
        'accuracy': [],
        # Cosine distance between two arrays of compared words
        # Range: [0, 1] - high is more similar
        'cosine': [],
        # Average cosine distance of reference model words to query word
        'ref_cosine_query': [],
        # Average cosine distance of algorithm 1 words to query word
        'algo1_cosine_query': [],
        # Average cosine distance of algorithm 2 words to query word
        'algo2_cosine_query': [],
        # Word Mover distance between two arrays of compared words
        # Range: [0, 1] - high is more similar
        'wordmover': []
    }

    with open(f'{words_path}/{query}_words.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Compare each algorithm to embedding model words
    for model in ref_models:
        if len(data[model]) < 75:
            continue
        
        for algo in algos:
            df['name'].append(f'{model}-{algo}')
            df['IoU'].append(iou(data[model], data[algo]))
            df['advIoU'].append(advancediou(data[model], data[algo]))
            df['posIoU'].append(positionaliou(data[model], data[algo]))
            df['accuracy'].append(accuracy(data[model], data[algo]))
            cos, cos_query1, cos_query2 = cosine_distance(data[model], data[algo], query=query, model=ref_models[model].model)
            df['cosine'].append(cos)
            df['ref_cosine_query'].append(cos_query1)
            df['algo1_cosine_query'].append(cos_query2)
            df['algo2_cosine_query'].append(None)
            df['wordmover'].append(wm_distance(data[model], data[algo], model=ref_models[model].model))

    # Compare each algorithm to other algorithm
    for algo1 in algos:
        for algo2 in algos:
            if algo1 == algo2 or f'{algo2}-{algo1}' in df['name'] or len(data[algo2]) < 75 or len(data[algo1]) < 75:
                continue
            df['name'].append(f'{algo1}-{algo2}')
            df['IoU'].append(iou(data[algo1], data[algo2]))
            df['advIoU'].append(advancediou(data[algo1], data[algo2]))
            df['posIoU'].append(positionaliou(data[algo1], data[algo2]))
            df['accuracy'].append(accuracy(data[algo1], data[algo2]))

            avg_cos_query = []
            avg_cos1_query = []
            avg_cos2_query = []
            avg_wm = []

            # Compare average cosine distances to query for each algo with respect to every embedding model
            for md in ref_models:
                cos, cos_query1, cos_query2 = cosine_distance(data[algo1], data[algo2], query=query, model=ref_models[md].model)
                wm_dis = wm_distance(data[algo1], data[algo2], model=ref_models[md].model)
                avg_cos_query.append(cos)
                avg_cos1_query.append(cos_query1)
                avg_cos2_query.append(cos_query2)
                avg_wm.append(wm_dis)

            df['cosine'].append(np.average(avg_cos_query))
            df['ref_cosine_query'].append(None)
            df['algo1_cosine_query'].append(np.average(avg_cos1_query))
            df['algo2_cosine_query'].append(np.average(avg_cos2_query))
            df['wordmover'].append(np.average(wm_dis))

    dfp = pd.DataFrame.from_dict(df)

    # Save to csv
    dfp.to_csv(f'{result_path}/{query}_results.csv', encoding='utf-8', float_format='%.3f', na_rep=np.nan, index=False)