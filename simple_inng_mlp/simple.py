#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:14:38 2018

@author: zhanghuangzhao
"""

import gzip, pickle
import numpy
import random

class Simple(object):
    
    def __init__(self, path="./simple.pkl.gz", rand_seed=1234):
        
        with gzip.open(path, "rb") as f:
            self.__d = pickle.load(f)
            self.__d_neg = []
        
        random.seed(rand_seed)
        self.__epoch = random.sample(range(len(self.__d)), len(self.__d))
        self.__size_ori = len(self.__d)
        self.__size_neg = len(self.__d_neg)
        self.__size = len(self.__d)
        
    def minibatch(self, bs):
        
        if bs > self.__size:
            bs = self.__size
        if len(self.__epoch) < bs:
            self.__epoch = random.sample(range(self.__size), self.__size)
            
        idx = self.__epoch[:bs]
        self.__epoch = self.__epoch[bs:]
        
        b = [[], []]
        for i in idx:
            if i < self.__size_ori:
                b[0].append(self.__d[i])
                b[1].append(1)
            else:
                b[0].append(self.__d_neg[i - self.__size_ori])
                b[1].append(0)
        b[0] = numpy.asarray(b[0], dtype=numpy.float32)
        b[1] = numpy.asarray(b[1], dtype=numpy.float32)
        return b
    
    def set_neg(self, neg_data):
        
        self.__d_neg = numpy.asarray(neg_data, dtype=numpy.float32)
        self.__size = len(self.__d) + len(self.__d_neg)
    
    def get_size(self):
        
        return self.__size
    
    
if __name__ == "__main__":
    
    data = Simple()
    for i in range(1000):
        b = data.minibatch(32)