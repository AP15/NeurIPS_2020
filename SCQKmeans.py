# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:09:03 2020

@author: apaudice
"""

import numpy as np
import Clustering
import BinSearch

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
        X : {array-like}, shape = [#points, #features]
            Points to cluster.
        y : {array-like}, shape = [#points]
            Cluster membership oracle.
        k : int
            #Clusters
        C_star : Clustering object
            Ground truth clustering
 
        Returns
        ------------
        C : list
            Clustering of X.
        """
        
        C = {} 
        S = X 
        l = k*100
        searcher = BinSearch()
        
        #for i in np.arange(k):
        #Phase 1
        Z, labels = C_star.sample(l)
        unique, counts = np.unique(Z, return_counts=True)
        p = np.argmax(counts)
        mu_p = np.mean(Z[np.where(labels == p)], axis = 0)
        
        #Phase 2
        dist_mu = np.linalg.norm(C_star.X - np.tile(mu_p, (C_star.X.shape[0], 1)), axis = 1)
        dist_sort = np.sort(dist_mu)
        y_sort = C.y[np.argsort(np.argsort(dist_mu))]        
        idx = searcher.find(y_sort, p)
        r = dist_sort[idx]        
        
        C = C_start.X[np.where(dist_mu <= r)]
        
        
        return C
    
# =============================================================================
#     def binarySearch(self, mu, C, label):
#         """Find ray r for a cluster mu via binary search.
#  
#         Parameters
#         ------------
#         mu : {array-like}, shape = [#features]
#                 Estimated center of a cluster.
#         C : Clustering object
#             Ground truth clustering
#  
#         Returns
#         ------------
#         radius : float
#             Ray of cluster mu.
#         """
#         
#         #Compute distances from mu
#         dist_mu = np.linalg.norm(Z - np.tile(mu, (C.X.shape[0], 1)), axis = 1)
#         dist_sort = np.sort(dist_mu)
#         y_sort = C.y[np.argsort(np.argsort(dist_mu))]
#         
#         #Perform a binary search
#         b = 1
#         r = C.X.shape[0]
#         radius = -1
#         while (r >= b):
#         
#             mid = b + (r - b) // 2
#   
#             # If element is present at the middle itself 
#             if y_sort[mid] == label: 
#                 radius = dist_sort[mid]
#                 b = mid + 1
#           
#             # If element is smaller than mid, then it  
#             # can only be present in left subarray 
#             else: 
#                 r = mid-1 
#         
#         return radius
# =============================================================================
    
                