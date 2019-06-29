#!/usr/bin/env python3

'''
@Time   : 2019-06-29 11:58:55
@Author : su.zhu
@Desc   : 
'''

import torch
import numpy as np

class word_digit_features_extractor():

    def __init__(self, max_digit_number=5, device=None):
        self.max_digit_number = max_digit_number
        self.device = device

    def get_feature_dim(self,):
        return self.max_digit_number

    def get_digit_features(self, word_seqs, lens):
        one_hot_features = np.eye(self.max_digit_number)
        max_len = max(lens)
        feature_seqs = []
        for seq in word_seqs:
            features = []
            for word in seq:
                if not (set(word) - set('0123456789')):
                    if len(word) < self.max_digit_number:
                        idx = len(word) - 1
                    else:
                        idx = self.max_digit_number - 1
                    features.append(one_hot_features[idx])
                else:
                    features.append([0] * (self.max_digit_number))
            features += [[0] * (self.max_digit_number)] * (max_len - len(seq))
            feature_seqs.append(features)
        return torch.tensor(feature_seqs, dtype=torch.float, device=self.device)
