# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:49:16 2020

@author: apaudice
"""

import numpy as np
import datagen

# Sample usage
import matplotlib.pyplot as plt
print("generating clusters...")
X, y, Ws, cs = datagen.randomDataset(n=1000, k=5, d=2, r=2, gamma=.5, cn=10, 
                                  tightMargin=True)
fig = plt.figure(figsize=(5, 5))
fig.add_subplot(111).scatter(X[:, 0], X[:, 1], c=y, s=1)
if True: # plot each cluster in its latent space
    for i in np.unique(y):
        Xt = datagen.toLatent(X, Ws[i], cs[i])
        fig=plt.figure()
        ax=fig.add_subplot(111)
        C=Xt[y==i]
        ax.scatter(Xt[:, 0], Xt[:, 1], c=y, s=3)
        ax.set_xlim(4*C[:, 0].min(), 4*C[:, 0].max())
        ax.set_ylim(4*C[:, 1].min(), 4*C[:, 1].max())
plt.show()