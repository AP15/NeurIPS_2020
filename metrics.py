# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:22:20 2020

@author: apaudice
"""

import Clustering
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Evaluation(object):
    def __init__(self):
                
    def recError(self, C, data):
        X, y = C.clustersTodata()
        #accuracy_score(data.y_, y)