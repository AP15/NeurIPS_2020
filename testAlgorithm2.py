# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:06:40 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import datagen
import ellipsoidalClustering as ecc
import oracle

#Suppress Warnings
warnings.filterwarnings("ignore")

#np.random.seed(0)
n_p = 10000
n_k = 5
g = .5
dim = 15
rank = dim
X_, y_, Ws, cs = datagen.randomDataset(n=n_p, k=n_k, d=dim, gamma=g, 
                                       tightMargin=True)

plot = True
if dim==2 and plot:
    fig, ax1 = plt.subplots()
    ax1.scatter(X_[:,0], X_[:,1], s=5, c=y_)
    plt.show()

O = oracle.SCQOracle(pd.DataFrame(y_))

X = pd.DataFrame(X_) # Our unlabeled dataset
X['y'] = np.nan

# Test cluster
alg = ecc.ECC(n_k, g)
C, n_queries = alg.cluster(X, O)

print("#Queries: %d" % n_queries)
print("Accuracy: %2f" % (sum(C==y_)/n_p))