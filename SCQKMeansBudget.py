# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:19:38 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import BinSearch as bs
import matplotlib.pyplot as plt

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
 
    def __init__(self, k):
        self.k = k
    
    def cluster(self, data, oracle, B):       
        #Get active nodes      
        activeNodes = data.copy()
        n = activeNodes.shape[0]  
        C = np.nan*np.ones(n)
        searcher = bs.BinSearch()

        for i in np.arange(self.k):
            #Phase 1
            print('*********************************')
            print("%d points left to cluster." % n)
            #print(activeNodes)
            # 1. Sample
            l = int(min(n, 10 * self.k))
            print("1.", l, " Taking samples")
            S = activeNodes.sample(l).index
            S_labels = np.array([oracle.label(i) for i in S]) 
            if l==n: # we have labeled all of X
                C[S]=S_labels
                break
            C[S] = S_labels
            
            nc = pd.DataFrame({'idx': S, 'lab': S_labels}).groupby('lab').count() # number of samples per cluster
            p = nc.sort_values(by='idx',ascending=False).index[0] # id of the largest cluster sample
            S_C = S[S_labels==p] #Idx of largest cluster
            X_SC = activeNodes.loc[S_C] #Datapoints of the largest cluster
            print("Got %d points from cluster %d" % (len(S_C), p))
            #print('X_SC:\n', X_SC)
            mu_p = np.mean(X_SC.values, axis = 0)
            print('Mean:', mu_p)
            
            #Phase 2
            X = activeNodes.values
            dist_mu = np.linalg.norm(X - np.tile(mu_p, (n, 1)), axis = 1)
            sort_idxs = np.argsort(dist_mu)
            dist_sort = dist_mu[sort_idxs]
            #print('Distances:', dist_sort)
            idxs = activeNodes.index
            idxs = idxs[sort_idxs]
            #idxs = activeNodes.loc[sort_idxs].index
            #print('Target:', activeNodes.loc[S_C[0]].values)
            r = searcher.findRaySCQ(dist_sort, oracle, idxs, S_C[0]) 
            print('Ray:', r)
            
            #Form cluster
            cluster_p_idxs = activeNodes.loc[dist_mu <= r].index
            #print('Clustered points:', cluster_p_idxs)
            C[cluster_p_idxs] = p
            activeNodes = data.loc[np.isnan(C)]
            n = activeNodes.shape[0]
            
            if (oracle.getCount() > B):
                print('Budget exhausted.')
                break
        
        return C, oracle.getCount()
    
    
    def clusterMonitor(self, data, oracle, y):
        np.random.seed(0)
        queries_s, queries_m = 0, 0
        queries, scores = [], []
        #Get active nodes      
        activeNodes = data.copy()
        n = activeNodes.shape[0]  
        No = n
        C = np.nan*np.ones(n)
        searcher = bs.BinSearch()

        for i in np.arange(self.k):
            #Phase 1
            print('*********************************')
            print("%d points left to cluster." % n)
            #print(activeNodes)
            # 1. Sample
            l = int(min(n, 10 * self.k))
            print("1.", l, " Taking samples")
            S = activeNodes.sample(l).index
            S_labels = np.array([oracle.label(i) for i in S]) 
            if l==n: # we have labeled all of X
                C[S]=S_labels
                break
            C[S] = S_labels
            
            nc = pd.DataFrame({'idx': S, 'lab': S_labels}).groupby('lab').count() # number of samples per cluster
            p = nc.sort_values(by='idx',ascending=False).index[0] # id of the largest cluster sample
            S_C = S[S_labels==p] #Idx of largest cluster
            X_SC = activeNodes.loc[S_C] #Datapoints of the largest cluster
            print("Got %d points from cluster %d" % (len(S_C), p))
            #print('X_SC:\n', X_SC)
            mu_p = np.mean(X_SC.values, axis = 0)
            print('Mean:', mu_p)
            #print('Did', oracle.getCount() - queries_m, 'for sampling.')
            queries_s += oracle.getCount() - queries_m
            
            #Phase 2
            X = activeNodes.values
            dist_mu = np.linalg.norm(X - np.tile(mu_p, (n, 1)), axis = 1)
            sort_idxs = np.argsort(dist_mu)
            dist_sort = dist_mu[sort_idxs]
            #print('Distances:', dist_sort)
            idxs = activeNodes.index
            idxs = idxs[sort_idxs]
            #idxs = activeNodes.loc[sort_idxs].index
            #print('Target:', activeNodes.loc[S_C[0]].values)
            r = searcher.findRaySCQ(dist_sort, oracle, idxs, S_C[0]) 
            print('Ray:', r)
            
            #Print Heatmap
            if (False):
                f = plt.figure()      
                plt.scatter(X[:,0], X[:,1], c = dist_mu, s=50)    
                plt.scatter(mu_p[0], mu_p[1], color='r', marker = 'x', label='mu', s= 200, linewidths=20)
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.grid()
                plt.show()
            
                f = plt.figure()     
                plt.scatter(X[:,0], X[:,1], c = dist_mu, s=50)    
                plt.scatter(X[dist_mu <= r,0], X[dist_mu <= r,1], color='r', marker = '1', label='mu', s= 50, linewidths=20)
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.grid()
                plt.show()
            
            #Form cluster
            cluster_p_idxs = activeNodes.loc[dist_mu <= r].index
            #print('Clustered points:', cluster_p_idxs)
            C[cluster_p_idxs] = p
            activeNodes = data.loc[np.isnan(C)]
            n = activeNodes.shape[0]
            scores.append(sum(C==y)/No)
            queries.append(oracle.getCount())
            queries_m = oracle.getCount()
            
            #Print Remaining points
            if (False):
                f = plt.figure()
                plt.scatter(activeNodes.values[:, 0], activeNodes.values[:, 1],
                            s=10, 
                            marker='o')
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.grid()
                plt.show()
        
        queries_m -= queries_s
        queries_s = oracle.getCount() - queries_m
        print('Queries Sampling:', queries_s)
        print('Queries Cleaning:', queries_m)
        
        return C, oracle.getCount(), np.asarray(scores), np.asarray(queries)