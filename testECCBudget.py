# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:26:25 2020

@author: apaudice
"""

import warnings
import utility
import experiments as exp

#Suppress Warnings
warnings.filterwarnings("ignore")


n_exp = 5
rep = 3

n = 1000000
d = 2
k = 10

experiments = exp.Experiments()

X, y, n, k = experiments.dataGeneration('parallel', n, d, k, gamma=1, rank=2, cn=50)

experiments.expAccuracyBudget(dataset = 'parallel',
                              algorithm = 'ecc', 
                              X_data = X,
                              y_data = y,
                              n = n,
                              d = d,
                              k = k,
                              n_exp = n_exp,
                              rep = rep)

experiments.expAccuracyBudget(dataset = 'parallel',
                              algorithm = 'kmeans', 
                              X_data = X,
                              y_data = y,
                              n = n,
                              d = d,
                              k = k,
                              n_exp = n_exp,
                              rep = rep)

utility.plotAccuracyBudget('parallel')