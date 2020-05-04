# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:19:11 2020

@author: apaudice
"""

import numpy as np
import qinfer.utils as ut
from sklearn.datasets import make_blobs
from sklearn.datasets import make_spd_matrix
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy.linalg as la

d = 2

#Data creation
X, y = make_blobs(n_samples=100, 
          n_features=d, 
          centers=1, 
          cluster_std=1, 
          shuffle=True)
          #random_state=0) 

mu = np.random.normal(0, 1, d)
cov = make_spd_matrix(d)
X = np.random.multivariate_normal(mu, cov, 100)

A, c = ut.mvee(X)

U, D, V = la.svd(A)

# x, y radii.
rx, ry = 1./np.sqrt(D)
# Major and minor semi-axis of the ellipse.
dx, dy = 2 * rx, 2 * ry
a, b = max(dx, dy), min(dx, dy)
# Eccentricity
e = np.sqrt(a ** 2 - b ** 2) / a

arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
arccos = np.rad2deg(np.arccos(V[0][1]))
alpha = arccos if arcsin > 0. else -1. * arccos


# Plot.
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')


# Plot ellipsoid.
ax = plt.gca()
ellipse2 = Ellipse(xy=c, width=a, height=b, edgecolor='k',
    angle=alpha, fc='None', lw=2)
ax.add_patch(ellipse2)

# Plot points.
plt.scatter(X[:, 0], X[:, 1],
    s=50, 
    c='lightgreen', 
    marker='s', 
    edgecolor='black',
    label='cluster 1')
# Plot center.
plt.scatter(*c,
             s=500, 
             marker='*',
             c='red', 
             edgecolor='black',
             label='centroids')

plt.show()