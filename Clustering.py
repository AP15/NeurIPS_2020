# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:35:24 2020

@author: apaudice
"""

import numpy as np
import Dataset

class Clustering(object):
    """Clustering of a set of points.
    
    Parameters
    ------------
    X : {array-like}, shape = [#points, #features]
        Points to cluster.
    y : {array-like}, shape = [#oints]
        Cluster memberiships.
    k : int
        Maximum #clusters.

    Attributes
    ------------
    n_ : int
        #points clustered.
    k_count_ : int
        #clusters actually in the clustering. 
    sizes_ : {array-like}, shape = [#clusters]
        Sizes of the clusters.
    probabilities_ : {arra-like}, shape = [#clusters]
        Relative (to the total number of points) sizes of the clusters.
    """
    
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        
        self.k_count_ = k
        self.n_ = X.shape[0]
        self.k_list_ = {}
        self.sizes_ = np.zeros(k)
        self.probabilities_ = np.zeros(k)

        for i in np.arange(k):
            self.sizes_[i] = np.count_nonzero(y == i)
            self.probabilities_[i] = self.sizes_[i]/y.size
            self.k_list_[i] = np.argwhere(y == i).tolist()
                        
    
    def addCluster(self, C, label):
        """Add a cluster to the clustering.
        
        Parameters
        ------------
        m : int
            Sample size.
        
        Returns
        ------------
        sample : {array-like}, shape = [#points, #features]
        labels
        """
        
        if (self.k_count_ >= self.k):
            raise NameError('Too many clusters!')
        else:
            self.k_list_[label] = C
            self.k_count_ = self.k_count_ + 1
            
            
    def removePoints(self, idx_list, label):
        if (self.k_count_ == 0):
            raise NameError('There are no clusters')
        elif (idx_list.size > self.k_list_[label].size):
            raise NameError('Too many points to remove.')
        else:
            self.k_list_[label] = np.delete(self.k_list_[label], idx_list)
            if (self.k_list_[label == 0]):
                self.k_count_ = self.k_count_ - 1
        
        
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
        
        idxs = np.random.choice(self.n_, m)
        sample = self.X[idxs]
        labels = self.y[idxs]
        
        return sample, labels
    
    def printClusters(self):
        print('There are: ', self.k_count_, ' clusters.')
        print(self.k_list_)

#Tests
data = Dataset.Dataset(n=10, d=2)
data.generate()

#Test: __init__
C = Clustering(data.X_, data.y_, 3)

#Test: sample
print('Test sample.')
print('------------\n')
Z, y = C.sample(5)
print('Samples.\n', Z)
print('Labels:', y, '\n')

print('Mean computation.')
print('------------\n')
unique, counts = np.unique(y, return_counts=True)
print('Sampled clusters: ', unique) 
print('#points per cluster: ', counts)

p = np.argmax(counts)
print('Largest sampled cluster:', p)
print(Z[np.where(y == p)])
print('Mean:', np.mean(Z[np.where(y == p)], axis = 0), '\n')

#Test for distances
print('Distance computation.')
print('------------\n')

mu = np.mean(Z[np.where(y == p)], axis = 0)
n = Z.shape[0]
print('Distances from the samples to mean:')
print(np.linalg.norm(Z - np.tile(mu, (n, 1)), axis = 1))