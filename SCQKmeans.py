# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:09:03 2020

@author: apaudice
"""

import numpy as np
import Dataset as ds
import Clustering
import BinSearch as bs

class SCQKmeans(object):
    """SBD algorithms for k-means with same-cluster-queries.
 
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
 
    def cluster(self, data, k_):
        """Scatter data along the first 2 features.
 
        Parameters
        ------------
        X : {array-like}, shape = [#points, #features]
            Points to cluster.
        y : {array-like}, shape = [#points]
            Cluster membership oracle.
        k_ : int
            #Clusters
 
        Returns
        ------------
        C : clustering
            Clustering of X.
        """

        searcher = bs.BinSearch()
        
        C = Clustering.Clustering(k = k_)
        
        l = k_*30
        
        for i in np.arange(k_):
            #Phase 1
            Z, labels = data.sample(l)
            unique, counts = np.unique(labels, return_counts=True)
            p = unique[np.argmax(counts)]
            mu_p = np.mean(Z[np.where(labels == p)], axis = 0)
            
            #Phase 2
            dist_mu = np.linalg.norm(data.X_ - np.tile(mu_p, (data.X_.shape[0], 1)), axis = 1)
            dist_sort = np.sort(dist_mu)
            y_sort = data.y_[np.argsort(np.argsort(dist_mu))]        
            r = searcher.findRay(dist_sort, y_sort, p) 
            
            #print(dist_mu)
            C.editCluster(data.getPoints(np.argwhere(dist_mu <= r)).tolist(), p)
            data.removePoints(np.argwhere(dist_mu <= r))
            
            if (data.n == 0):
                break
            
        if (data.n_ > 0):
            #self.clusterCompletion(data, C)
            C.declareUnclusterd(data.points_)
        
        return C
    
    
# =============================================================================
#     def clusteringCompletion(self, data, C):
#         n = data.n
# =============================================================================
        

#Tests
data = ds.Dataset(n=150, d=2, k=3)
data.generate()

alg = SCQKmeans()
C = alg.cluster(data, 3)
C.printClusters()