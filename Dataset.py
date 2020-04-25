# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:28:04 2020

@author: apaudice
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Dataset(object):
    """Synthetic Data generated from a mixture of gaussian.
    
    Parameters
    ------------
    n : int
        #Points.
    d : int
        #Features.
    k : int
        #Clusters.
    seed : int 
        Random number generator seed for sampling.

    Attributes
    ------------
    X_ : {array-like}, shape = [#Points, #Features]
        Data Matrix.
    y_ : {array-like}, shape = [#Points]
        Cluster membership oracle.
    """
    
    def __init__(self, n=150, d=2, k=3, seed=0):
        self.n = n
        self.d = d
        self.k = k
        self.seed = seed
    
    def generate(self):
        """Generate the data
        
        Parameters
        ------------
        None
        
        Returns
        ------------
        self : object
        """
        
        self.X_, self.y_ = make_blobs(n_samples=self.n, 
                  n_features=self.d, 
                  centers=self.k, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=self.seed)
        self.points_ = np.arange(self.n)
        
        return self


    def getPoints(self, idx_list):
        if (idx_list.size > self.n):
            raise NameError('Too many points to remove.')
        else:
            return self.points_[idx_list]
            
            
    def removePoints(self, idx_list):
        if (idx_list.size > self.n):
            raise NameError('Too many points to remove.')
        else:
            self.X_ = np.delete(self.X_, idx_list, 0)
            self.y_ = np.delete(self.y_, idx_list)
            self.points_ = np.delete(self.points_, idx_list)
            self.n -= idx_list.size

    
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
        sample = self.X_[idxs]
        labels = self.y_[idxs]
        
        return sample, labels
    
    def scatterData(self):
        """Scatter data along the first 2 features.
        
        Parameters
        ------------
        None
        
        Returns
        ------------
        None
        """
        
        plt.figure()
        plt.scatter(self.X_[self.y_ == 0, 0], self.X_[self.y_ == 0, 1],
            s=50, 
            c='lightgreen', 
            marker='s', 
            edgecolor='black',
            label='cluster 1')

        plt.scatter(self.X_[self.y_ == 1, 0], self.X_[self.y_ == 1, 1], 
            s=50, 
            c='orange', 
            marker='o', 
            edgecolor='black',
            label='cluster 2')

        plt.scatter(self.X_[self.y_ == 2, 0], self.X_[self.y_ == 2, 1],
            s=50, 
            c='lightblue',
            marker='v', 
            edgecolor='black',
            label='cluster 3')

        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.grid()
        plt.show()

# =============================================================================
# #Test: sample
# data = Dataset(n=10, d=2)
# data.generate()
# 
# print('Test sample.')
# print('------------\n')
# 
# Z, y = data.sample(5)
# 
# print('Samples.\n', Z)
# print('Labels:', y, '\n')
# 
# print('Mean computation.')
# print('------------\n')
# 
# unique, counts = np.unique(y, return_counts=True)
# 
# print('Sampled clusters: ', unique) 
# print('#points per cluster: ', counts)
# 
# p = np.argmax(counts)
# 
# print('Largest sampled cluster:', p)
# print(Z[np.where(y == p)])
# print('Mean:', np.mean(Z[np.where(y == p)], axis = 0), '\n')
# 
# #Test: removePoints
# #Test: sample
# print('Test removePoints.')
# print('------------\n')
# 
# print('Data:')
# print(data.X_, '\n')
# 
# idx = np.array([0, 1])
# data.removePoints(idx)
# 
# print('New data:')
# print(data.X_)
# =============================================================================

# =============================================================================
# #Find clusering
# km = KMeans(n_clusters=3, 
#             init='random',
#             n_init=10,
#             max_iter=300,
#             tol=1e-04,
#             random_state=0)
# 
# y_km = km.fit_predict(data.X_)
# 
# plt.figure()
# plt.scatter(data.X_[y_km == 0, 0], data.X_[y_km == 0, 1],
#             s=50, 
#             c='lightgreen', 
#             marker='s', 
#             edgecolor='black',
#             label='cluster 1')
# 
# plt.scatter(data.X_[y_km == 1, 0], data.X_[y_km == 1, 1], 
#             s=50, 
#             c='orange', 
#             marker='o', 
#             edgecolor='black',
#             label='cluster 2')
# 
# plt.scatter(data.X_[y_km == 2, 0], data.X_[y_km == 2, 1],
#             s=50, 
#             c='lightblue',
#             marker='v', 
#             edgecolor='black',
#             label='cluster 3')
# 
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
#             s=250, 
#             marker='*',
#             c='red', 
#             edgecolor='black',
#             label='centroids')
# 
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()
# =============================================================================

