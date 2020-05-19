# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:10:03 2020

@author: apaudice
"""

import numpy as np
from bisect import bisect_right 
import Dataset

class BinSearch(object):
    """Binary search for finding the rightmost accurence of a given element
    in an array.
    
    Parameters
    ------------
    None
    
    Attributes
    ------------
    None
    """
    
    def __init__(self):
        pass
    
    def find(self, a, x):
        i = bisect_right(a, x) 
        if i != len(a)+1 and a[i-1] == x: 
            return (i-1) 
        else: 
            return -1
    
    def findRay(self, dist, labels, y):
        
        l = 0
        r = dist.size-1
        radius = dist[0]
        # Check base case 
        while(r >= l): 
  
            mid = l + (r - l)//2
  
            # If element is present at the middle itself 
            if labels[mid] == y: 
                radius = dist[mid]
                l = mid+1
          
            # If element is smaller than mid, then it can only 
            # be present in left subarray 
            else: 
                r = mid-1   
        
        return radius
    
    
    def findRaySCQ(self, dist, oracle, idxs, cluster_id):
        
        l = 0
        r = dist.size-1
        radius = dist[0]
        # Check base case 
        while(r >= l): 
  
            mid = l + (r - l)//2
  
            # If element is present at the middle itself 
            #print(idxs[mid], cluster_id)
            if oracle.scq(idxs[mid], cluster_id): 
                radius = dist[mid]
                l = mid+1
          
            # If element is smaller than mid, then it can only 
            # be present in left subarray 
            else: 
                r = mid-1   
        
        return radius

       
# =============================================================================
# #Tests
# searcher = BinSearch()
# 
# arr = np.array([1, 1, 1, 1, 2, 3, 4, 5])
# x = 5
# 
# print('Array: ', arr)
# print('Find position of the last occurence of ' + str(x) + '.')
# print(searcher.find(arr, x))
# 
# #Test: find
# data = Dataset.Dataset(n=10, d=2)
# data.generate()
# 
# #Test: sample
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
# #Test for distances
# print('Distance computation.')
# print('------------\n')
# 
# mu = np.mean(Z[np.where(y == p)], axis = 0)
# n = data.X_.shape[0]
# 
# dist = np.linalg.norm(data.X_ - np.tile(mu, (n, 1)), axis = 1)
# dist_sort = np.sort(dist)
# y_sort = data.y_[np.argsort(np.argsort(dist))]        
# 
# print('Distances from the samples to mean:')
# 
# print(dist_sort)
# print(y_sort)
# 
# idx = searcher.find(y_sort, p)
# r = dist_sort[idx]
# 
# print(r)
# =============================================================================
