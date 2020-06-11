# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:01:06 2020

@author: apaudice
"""

import warnings
import utility
import experiments as exp
import numpy as np

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white

#Suppress Warnings
warnings.filterwarnings("ignore")

rep = 1

n = 100000
d = 2
k = 5
gamma = 1

data='general'

experiments = exp.Experiments()

X, y, n, k = experiments.dataGeneration(data, n, d, k, gamma=gamma, rank=d, cn=100)
np.savez('Data' + "_" + str(d), X, y)

X_white = svd_whiten(X)
np.savez('White_Data' + "_" + str(d), X_white, y)

# =============================================================================
# idxs = np.random.choice(n, 100000, False)
# utility.plotClustering(X_white[idxs, :], k, y[idxs], 'general')
# =============================================================================

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

experiments.expAccuracyQueries(dataset = data,
                              algorithm = 'white_kmeans', 
                              X_data = X_white,
                              y_data = y,
                              n = n,
                              d = d,
                              k = k,
                              gamma = gamma,
                              rep = rep)

utility.plotAccuracyQueriesPCA2(data, d)