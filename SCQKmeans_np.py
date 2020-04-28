# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:28:22 2020

@author: apaudice
"""

import numpy as np
import Dataset as ds
import BinSearch as bs
import copy

class SCQKmeansNP(object):
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
 
    def cluster(self, data, k_, B):
        """Scatter data along the first 2 features.
 
        Parameters
        ------------
        data : object
            Data to cluster.
        k_ : int
            #Clusters.
            
        B: int
            #Queries.
    
        Returns
        ------------
        C : clustering
            Clustering of data.
        """

        searcher = bs.BinSearch()
        
        C = -1*np.ones(data.n)
        
        l = k_*B
        
        for i in np.arange(k_):
            #Phase 1
            Z, labels, points = data.sample(l)
            unique, counts = np.unique(labels, return_counts=True)
            p = unique[np.argmax(counts)]
            mu_p = np.mean(Z[np.where(labels == p)], axis = 0)
            
            #Phase 2
            dist_mu = np.linalg.norm(
                    data.X_ - np.tile(mu_p, (data.X_.shape[0], 1)), axis = 1)
            sort_idxs = np.argsort(dist_mu)
            dist_sort = dist_mu[sort_idxs]
            y_sort = data.y_[sort_idxs]        
            r = searcher.findRay(dist_sort, y_sort, p) 
            
            C[data.getPoints(np.argwhere(dist_mu <= r))] = p
            data.removePoints(np.argwhere(dist_mu <= r))
            
            if (data.n == 0):
                break
        
        return C.astype(int)