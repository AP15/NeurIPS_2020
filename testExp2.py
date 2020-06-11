# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:50:11 2020

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
k = 5
gamma = 1

data='general'

# =============================================================================
# experiments = exp.Experiments()
# 
# X, y, n, k = experiments.dataGeneration(data, n, d, k, gamma=gamma, rank=d, cn=100)
# =============================================================================

# =============================================================================
# idxs = np.random.choice(n, 100000, False)
# utility.plotClustering(X[idxs, :], k, y[idxs], 'parallel')
# =============================================================================


# =============================================================================
# experiments.expAccuracyQueries(dataset = data,
#                               algorithm = 'ecc', 
#                               X_data = X,
#                               y_data = y,
#                               n = n,
#                               d = d,
#                               k = k,
#                               gamma = gamma,
#                               rep = rep)
# 
# experiments.expAccuracyQueries(dataset = data,
#                               algorithm = 'kmeans', 
#                               X_data = X,
#                               y_data = y,
#                               n = n,
#                               d = d,
#                               k = k,
#                               gamma = gamma,
#                               rep = rep)
# =============================================================================

utility.plotAccuracyQueries(data)