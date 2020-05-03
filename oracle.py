# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:14:26 2020

@author: apaudice
"""

import numpy as np

class SCQ0racle(object):
    """
    Same-Cluster-Query (SCQ) oracle.
    Given any pair (x,y) \in X, return True if C(x)=C(y) and False o.w.
    Counts also the number of asked SCQs so far.
    
    Attributes
    ----------
    data : pandas dataframe, shape = [#Points, #Features+1]
        Input data for clustering.
    dim : int
        #Features = #Columns of data.  
    count : int
        #SCQs asked so far.
    foundClusters: list, shape <= #Clusters in data
        Labels of the clusters for which at least a member has been found.
        
    Methods
    -------
    __init__(data : pandas dataframe)
        Construtor
    scq(idx : int, idy : int)
        Return SCQ answer for (idx, idy).
    label(idx : int)
        Return the label of idx.
    getCount()
        Return #SCQs asked so far.
        
    """
    
    def __init__(self, data):
        """
        Constructot.
        
        Parameters
        ----------
        data : pandas dataframe, shape = [#Points, #Features+1]
            Input data for clustering. Last column contains the labels.

        Returns
        -------
        None.

        """
        self.data = data.copy()
        self.dim = len(data.columns)
        self.count = 0
        self.foundClusters = []
        
    
    def scq(self, idx, idy):
        """
        Perform an SCQ on for the pair (x, y).
        Parameters
        ----------
        idx : int
            Id of a point x.
        idy : int
            Id of a point y.

        Returns
        -------
        boolean
            True if x and y have the same label, False otherwise.

        """
        self.count += 1
        return self.data[self.dim-1][idx] == self.data[self.dim-1][idy]
    
    
    def label(self, idx):
        """
        Perform a cluster membership query for x. count is updated
        as if the membership query would have been simulated by using SCQs.
        If the cluster of x has been already queried, then this query 
        counts as the index of label_x in foundClusters, otherwise
        it counts has len(foundClusters)

        Parameters
        ----------
        idx : int
            Id of a point x.

        Returns
        -------
        label_x : int
            Label of x.

        """
        label_x = self.data[self.dim-1][idx]
        if label_x in self.foundClusters:
            self.count += self.foundClusters.index(label_x)
        else:
            self.count += len(self.foundClusters)
            self.foundClusters.append(label_x)
            self.foundClusters.sort()
        return label_x
    
    
    def getCount(self):
        """
        Return the number of SCQs asked so far to the oracle.

        Returns
        -------
        int
            SCQs asked so far to the oracle.

        """
        return self.count
    
    