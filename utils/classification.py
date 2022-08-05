import numpy as np


def predict_cosine(audit_feature, feature_db):
    probability_list = []
    for feature in feature_db:
        if audit_feature.size == feature.size:
            probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
        else:
            probability = 0.0
        probability_list.append(probability)  
    # lấy ảnh có tỷ lệ giống cao nhất và so sánh với ngưỡng (THRESHOLD)
    max_prob = np.max(probability_list)
    max_index = probability_list.index(max_prob)
    return max_index, max_prob