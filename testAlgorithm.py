# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:06:40 2020

@author: apaudice
"""

import oracle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Dataset as data
import ellipsoidalClustering as ecc
import warnings

#Suppress Warnings
warnings.filterwarnings("ignore")


#np.random.seed(0)
n, d, k = 100, 3, 3
gamma = .2
ds = data.Dataset(n, d, k)
ds.generateEllipsoidsGeneral()

plot = True
if d==2 and plot:
    fig, ax1 = plt.subplots()
    ax1.scatter(ds.X_[:,0], ds.X_[:,1], s=5, c=ds.y_)
    plt.show()

O = oracle.SCQOracle(pd.DataFrame(ds.y_))

X = pd.DataFrame(ds.X_) # our unlabeled dataset
X['y']=np.nan

print(ds.y_)
print(X)

# Test cluster
alg = ecc.ECC(k, gamma)
C, n_queries = alg.cluster(X, O)

print("#Queries: %d" % n_queries)
print("Accuracy: %2f" % (sum(C==ds.y_)/n))