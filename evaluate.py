from evaluate_on_lfw import lfw
from utils.featureExtraction import feature_extraction
import cv2
import numpy as np


pair_filename_path = 'dataset/lfw/pairs.txt'
lfw_dir = 'dataset\\lfw\\lfw'
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

pairs = lfw.read_pairs(pair_filename_path)
paths, actual_issame = lfw.get_paths(lfw_dir, pairs)

print('Embedding on LFW images')
embeddings = []
for i in range(len(paths)):
    if i % 2 == 0: 
        img0 = cv2.imread(paths[i])
        embedding0 = feature_extraction(img0)
        img1 = cv2.imread(paths[i+1])
        embedding1 = feature_extraction(img1)
        embeddings += (embedding0, embedding1)

np.save('dataset/lfw/embedding.npy', embeddings)
# embeddings = np.load('dataset/lfw/embedding.npy')

for threshold in thresholds:
    print('---------------------------------------')

    accuracy, precision, recall = lfw.evaluate(embeddings, actual_issame, threshold)

    f1_score = 0 if (precision+recall==0) else 2*precision*recall/(precision+recall)
        
    print('Threshold: %1.2f' % threshold)

    print('Accuracy: %1.2f' % accuracy)
    print('Precision: %1.2f' % precision)
    print('Recall: %1.2f' % recall)
    print('F1 score: %1.2f' % f1_score)