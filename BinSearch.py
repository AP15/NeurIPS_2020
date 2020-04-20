# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:10:03 2020

@author: apaudice
"""

import numpy as np
from bisect import bisect_right 

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
 

       
#Tests
searcher = binSearch()

arr = np.array([1, 1, 1, 1, 2, 3, 4, 5])
x = 5

print('Array: ', arr)
print('Find position of the last occurence of ' + str(x) + '.')
print(searcher.find(arr, x))