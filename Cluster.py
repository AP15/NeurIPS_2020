# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:44:22 2020

@author: apaudice
"""

class Cluster(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_ = X.shape[0]