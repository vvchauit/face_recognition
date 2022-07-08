import  numpy as np


def calculate_accuracy(threshold, probs, actual_issame_list):
    correct = 0
    miss = 0
    incorrect = 0
    for i in range(len(probs)):
        if probs[i] >= threshold:
            if actual_issame_list[i]:
                correct += 1
            else:
                incorrect += 1
        else:
            miss += 1
    acc = correct / (correct + incorrect)
    val_rate = (correct + incorrect) / (correct + incorrect + miss)
    error_rate = (incorrect + miss) / (correct + incorrect + miss)
    return acc, val_rate, error_rate
    