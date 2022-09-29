import numpy as np
import math

def predict_cosine(audit_feature, feature_db):
    probability_list = []
    distance_list = [] = []
    for feature in feature_db:
        if audit_feature.size == feature.size:
            probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
            distance = np.arccos(probability) / math.pi
        else:
            probability = 0.0
            distance = 100.0
        probability_list.append(probability)
        distance_list.append(distance)
    # lấy ảnh có tỷ lệ giống cao nhất và so sánh với ngưỡng (THRESHOLD)
    max_prob = np.max(probability_list)
    max_index = probability_list.index(max_prob)
    min_dist = np.min(distance_list)
    min_index = probability_list.index(max_prob)
    return max_index, max_prob, min_dist, min_index