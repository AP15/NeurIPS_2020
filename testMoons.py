# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:08:56 2020

@author: apaudice
"""

import warnings
import utility
import experiments as exp
import numpy as np

#Suppress Warnings
warnings.filterwarnings("ignore")


rep = 10

n = 100000
d = 2
k = 2
gamma = 1

data='parallel'

experiments = exp.Experiments()

X, y, n, k = experiments.dataGeneration(data, n, d, k, gamma=gamma, rank=d, cn=100)

idxs = np.random.choice(n, 100000, False)
utility.plotClustering(X[idxs, :], k, y[idxs], 'parallel')

X_pca = utility.pcaData(X, k)

utility.plotClustering(X_pca[idxs, :], k, y[idxs], 'parallel')

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
                              X_data = X_pca,
                              y_data = y,
                              n = n,
                              d = d,
                              k = k,
                              gamma = gamma,
                              rep = rep)

utility.plotAccuracyQueriesPCA(data, d)
