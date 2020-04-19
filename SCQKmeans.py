# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:09:03 2020

@author: apaudice
"""

import numpy as np
import dataGeneration as dataG

class Clustering(object):
    """Clustering of a set of points.
    
    Parameters
    ------------
    y : {array-like}, shape = [#Points]
        Cluster memberiships.
    k : int
        #Clusters.

    Attributes
    ------------
    k_list_ : {arraylike}, shape = [#Clusters]
    sizes_ : {array-like}, shape = [#Clusters]
        Sizes of the clusters.
    probabilities_ : {arra-like}, shape = [#Clusters]
        Relative (to the total number of points) sizes of the clusters.
    """
    
    def __init__(self, y, k):
        self.y = y
        self.k = k
        self.k_list_ = np.arange(k)
        self.sizes_ = np.zeros(k)
        self.probabilities_ = np.zeros(k)
        for i in self.k_list_:
            self.sizes_[i] = np.count_nonzero(y == i)
            self.probabilities_[i] = self.sizes_[i]/y.size
    
    def sample(self, n):
        return np.random.choice(self.k, n, p=self.probabilities_)

    def sampleU(self, n):
        return np.random.choice(self.k, n)

data = dataG.Dataset()
data.generate()

C = Clustering(data.y_, 3)
print(C.sampleU(100))

class SCQKmeans(object):
    """Shai algorithms for k-means with same-cluster-queries.
 
    Parameters
    ------------
    l : int
    #Queries for Phase 1.
    delta : float in (0,1)
    Success probability.
    seed : int 
    Random number generator seed for randomness.
 
    Attributes
    ------------
    """
 
    def __init__(self, l=100, delta=0.95, seed=0):
        self.l = l
        self.delta = delta
        self.seed = seed
 
    def cluster(self, X, y, k, C_star):
        """Scatter data along the first 2 features.
 
        Parameters
        ------------
        X : {array-like}, shape = [n_points, n_features]
            Points to cluster.
        y : {array-like}, shape = [n_points]
            Cluster membership oracle.
        k : int
            #Clusters
        C : Clustering object
            Ground truth clustering
 
        Returns
        ------------
        C : list
            Clustering of X.
        """
 
        C = {} 
        S = X 
        l = k*100
 
        for i in np.arange(k):
            #Phase 1
            Z = C_star.sampleU(l)

