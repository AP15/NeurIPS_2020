# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:08:56 2020

@author: apaudice
"""

import warnings
import utility
import experiments as exp

#Suppress Warnings
warnings.filterwarnings("ignore")


rep = 10

n = 10000
d = 2
k = 3
gamma = 1

data='moons'

experiments = exp.Experiments()

X, y, n, k = experiments.dataGeneration(data, n, d, k, gamma=gamma, rank=d, cn=100)

experiments.expAccuracyQueries(dataset = data,
                              algorithm = 'ecc', 
                              X_data = X,
                              y_data = y,
                              n = n,
                              d = d,
                              k = k,
                              gamma = gamma,
                              rep = rep)

experiments.expAccuracyQueries(dataset = data,
                              algorithm = 'kmeans', 
                              X_data = X,
                              y_data = y,
                              n = n,
                              d = d,
                              k = k,
                              gamma = gamma,
                              rep = rep)

utility.plotAccuracyQueries(data)