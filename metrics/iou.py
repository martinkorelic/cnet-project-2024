def iou(array1, array2):

    set1 = set(array1)
    set2 = set(array2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if not union:
        return 0
    iou_score = len(intersection) / len(union)
    return iou_score