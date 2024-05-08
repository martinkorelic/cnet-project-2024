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