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
    U : {array-like}, shape = [., #features]
        Unclustered points.
    U_list :array-like}, shape = [., #features]
        Unclustered points.

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

    def __init__(self, y = np.array([]), k = 1):
        self.k = k
    
        self.k_count_ = 0
        self.n_ = y.size
        self.clusters_ = {i: [] for i in range(k)}
        self.sizes_ = np.zeros(k)
        self.probabilities_ = np.zeros(k)
        self.U_list = []
    
        if (y.size > 0):
            for i in np.arange(k):
                self.k_count_ += 1
                self.sizes_[i] = np.count_nonzero(y == i)
                self.probabilities_[i] = self.sizes_[i]/y.size
                self.clusters_[i] = np.argwhere(y == i).tolist()
                
    def areUnlustered(self):
        if (len(self.U_list) > 0):
            return True
        else:
            return False
        
    def declareUnclusterd(self, idx):
        self.U_list = idx
    
    def editCluster(self, C, label):
        if ((len(self.clusters_[label]) == 0)):
            self.k_count_ += 1
        
        #☺temp = self.clusters_[label]        
        #temp += C
        #self.clusters_[label] = list(dict.fromkeys(temp))
        self.clusters_[label] += C 
        self.sizes_[label] += len(C)
        self.n_ += len(C)
        self.probabilities_[label] = self.sizes_[label]/self.n_
        
    def clustersTodata(self):
        X = np.array([])
        y = np.array([])
        for i in np.arange(self.k_count_):
            X = np.concatenate((X, np.array(self.clusters_[i])), axis = 0)
            y = np.concatenate((y, i * np.ones(len(self.clusters_[i]))), axis = 0)
        X = np.concatenate((X, np.array(self.U_list)), axis = 0)
        y = np.concatenate((y, -1 * np.ones(len(self.U_list))), axis = 0)
        idx = np.argsort(X)
        
        return X[idx], y[idx]
        
    
    def printClusters(self):
        print('There are: ', self.k_count_, ' clusters.')
        print(self.clusters_)


#Tests
data = Dataset.Dataset(n=150, d=2)
data.generate()

#Test: __init__
# =============================================================================
# C = Clustering(k = 3)
# 
# C.printClusters()
# 
# C.editCluster(np.argwhere(data.y_ == 1).tolist(), 1)
# 
# C.printClusters()
# 
# C.editCluster(np.argwhere(data.y_ == 1).tolist(), 1)
# C.printClusters()
# =============================================================================

#Test: clusterTodata
C1 = Clustering(data.y_, k = 3)
C1.printClusters()
X, y = C1.clustersTodata()