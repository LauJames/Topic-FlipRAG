import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))  # Method to determine the root directory
import torch
from torch import cuda
import json
import bisect
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from apex import amp
import logging


# from adv_ir.attack_methods import pairwise_anchor_trigger
from sklearn.metrics import ndcg_score

device = 'cuda' if cuda.is_available() else 'cpu'

def topk_proportion(original_label_rank, later_label_rank, polarity, topk = [3, 6]):
    result = {}
    for k in topk:
        target = original_label_rank[:k]
        score = len([t for t in target if t == polarity])/k
        print("Proportion of", polarity, "in the original top", k, ":", score)
        target_2 = later_label_rank[:k]
        score_2 = len([t for t in target_2 if t == polarity])/k
        print("Proportion of", polarity, "in the attacked top", k, ":", score_2)
        result['before-top'+str(k)] = score
        result["later-top"+str(k)] = score_2
        print("Improvement:", score_2 - score)
        result["top"+str(k)+"-improvement"] = score_2 - score
    return result

def topk_proportion_to_length(original_label_rank, later_label_rank, polarity, topk):
    result = {}
    for k in topk:
        target = original_label_rank[:k]
        score = len([t for t in target if t == polarity])/k
        print("Proportion of", polarity, "in the original top 1/2:", score)
        target_2 = later_label_rank[:k]
        score_2 = len([t for t in target_2 if t == polarity])/k
        print("Proportion of", polarity, "in the attacked top 1/2:", score_2)
        result['before-top1/2'] = score
        result["later-top1/2"+str(k)] = score_2
        print("Improvement:", score_2 - score)
        result["top1/2-improvement"] = score_2 - score
    return result

def topk_mutual_score(original_label_rank, later_label_rank, polarity, topk = [3, 6]):
    # Evaluate the relative proportion of non-target labels in the top-k: success improvement/possible positions
    result = {}
    for k in topk:
        if k == 1/2:
            k = int(len(original_label_rank)/2)
            name = "1/2"
        else:
            name = k
        target = original_label_rank[:k]
        score = len([t for t in target if t != polarity])/k
        print("Proportion of non-", polarity, "in the original top", k, ":", score)
        target_2 = later_label_rank[:k]
        score_2 = len([t for t in target_2 if t != polarity])/k
        print("Proportion of non-", polarity, "in the attacked top", k, ":", score_2)
        if score == 0:
            score_2 = 1
            score = 1
        result["later-top-not-proportion"+str(name)] = score_2
        print("Relative improvement for top", name, ":", 1 - (score_2 / score))
        result["top"+str(name)+"-relative-improvement"] = 1 - (score_2 / score)

    return result

def recall_score(original_label_rank, later_label_rank, polarity):
    # Evaluate the proportion of the target label in the top-n (similar to recall)
    result = {}
    target_num = len([t for t in original_label_rank if t == polarity])
    target = original_label_rank[:target_num]
    score = len([t for t in target if t == polarity]) / target_num
    print("Proportion of", polarity, "in the original top", target_num, ":", score)
    target_2 = later_label_rank[:target_num]
    score_2 = len([t for t in target_2 if t == polarity]) / target_num
    print("Proportion of", polarity, "in the attacked top", target_num, ":", score_2)
    if score == 0:
        score_2 = 1
        score = 1
    print("Recall improvement for top", target_num, ":", score_2 - score)
    result["recall-improvement"] = score_2 - score
    return result

def avg_rank_boost(original_label_rank, later_label_rank, polarity):
    origin_rank_sum = sum([t for t in range(len(original_label_rank)) if original_label_rank[t] == polarity])
    later_rank_sum = sum([t for t in range(len(later_label_rank)) if later_label_rank[t] == polarity])
    boost_sum = origin_rank_sum - later_rank_sum
    avg_boost_rank = boost_sum / len(later_label_rank)
    return avg_boost_rank, boost_sum, len([t for t in original_label_rank if t == polarity])

def relabel_polarity(tag, labels):
    for t in range(len(labels)):
        if labels[t] == tag:
            labels[t] = 3
        elif labels[t] == 2:
            labels[t] = 1
        else:
            labels[t] = 0
    return labels

def cal_NDCG(scores, labels, k=10):
    scores = np.array([scores])
    labels = np.array([labels])
    ndcg = ndcg_score(labels, scores, k=k)
    return ndcg