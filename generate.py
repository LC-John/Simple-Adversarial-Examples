#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:56:48 2018

@author: zhanghuangzhao
"""

import numpy
import gzip, pickle
import matplotlib.pyplot as plt

import tensorflow as tf

def generate_square(size, x=0, y=0, a=10):
    
    x = numpy.random.uniform(low=x-a, high=x+a, size=size)
    y = numpy.random.uniform(low=y-a, high=y+a, size=size)
    res = numpy.asarray((x, y)).transpose((1, 0))
    
    return res

if __name__ == "__main__":
    
    d1 = generate_square(1000, 2, 2, 1)
    d3 = generate_square(1000, -2, -2, 1)
    
    plt.plot(d1[:, 0], d1[:, 1], '.')
    plt.plot(d3[:, 0], d3[:, 1], '.')
    
    d = numpy.concatenate([d1, d3], 0)
    
    with gzip.open("./simple.pkl.gz", "wb") as f:
        pickle.dump(d, f)