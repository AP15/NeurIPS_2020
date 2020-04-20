# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:35:24 2020

@author: apaudice
"""

import numpy as np
import dataGeneration

class Clustering(object):
    """Clustering of a set of points.
    
    Parameters
    ------------
    X : {array-like}, shape = [#points, #features]
        Points to cluster.
    y : {array-like}, shape = [#oints]
        Cluster memberiships.
    k : int
        #Clusters.

    Attributes
    ------------
    sizes_ : {array-like}, shape = [#clusters]
        Sizes of the clusters.
    probabilities_ : {arra-like}, shape = [#clusters]
        Relative (to the total number of points) sizes of the clusters.
    """
    
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        
        self.n = X.shape[0]
        self.sizes_ = np.zeros(k)
        self.probabilities_ = np.zeros(k)
        for i in self.k_list_:
            self.sizes_[i] = np.count_nonzero(y == i)
            self.probabilities_[i] = self.sizes_[i]/y.size
    
    def addCluster(self, C):
        
    
    def sample(self, m):
        """Sample point u.a.r. from X. Return both the points and the labels.
        
        Parameters
        ------------
        m : int
            Sample size.
        
        Returns
        ------------
        sample : {array-like}, shape = [#points, #features]
        labels : {array-like}, shape = [#points]
        """
        
        idxs = np.random.choice(self.n, m)
        sample = self.X[idxs]
        labels = self.y[idxs]
        
        return sample, labels

#Tests
data = dataGeneration.Dataset(n=10, d=2)
data.generate()

#Test: __init__
C = Clustering(data.X_, data.y_, 3)

#Test: sample
Z, y = C.sample(5)
print(Z, y)

unique, counts = np.unique(y, return_counts=True)
print(unique, counts)
p = np.argmax(counts)
print(p)
print(Z[np.where(y == p)])
print('Stop.')
print('Mean:\n', np.mean(Z[np.where(y == p)], axis = 0))

#Test for distances
mu = np.mean(Z[np.where(y == p)], axis = 0)

n = Z.shape[0]
#print(np.tile(mu, (n, 1)))
print(np.linalg.norm(Z - np.tile(mu, (n, 1)), axis = 1))