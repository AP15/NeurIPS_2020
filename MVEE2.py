# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:19:11 2020

@author: apaudice
"""

import numpy as np
import qinfer.utils as ut
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy.linalg as la

#Data creation
X, y = make_blobs(n_samples=100, 
          n_features=2, 
          centers=1, 
          cluster_std=0.5, 
          shuffle=True)
          #random_state=0) 

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
# Orientation angle (with respect to the x axis counterclockwise).
alpha = arccos if arcsin > 0. else -1. * arccos
# print -1*np.rad2deg(np.arcsin(V[0][0])), np.rad2deg(np.arccos(V[0][1]))
# print np.rad2deg(np.arccos(V[1][0])), np.rad2deg(np.arcsin(V[1][1]))


# Plot.
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

# u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]
# x0 = rx * np.cos(u) * np.cos(v)
# y0 = ry * np.sin(u) * np.cos(v)
# E = np.dstack([x0, y0])
# E = np.dot(E, V) + centroid
# x1, y1 = np.rollaxis(E, axis=-1)
# ax.plot(x1, y1)

# Plot ellipsoid.
ax = plt.gca()
ellipse2 = Ellipse(xy=c, width=a, height=b, edgecolor='k',
    angle=alpha, fc='None', lw=2)
ax.add_patch(ellipse2)

# Plot points.
plt.scatter(X[:, 0], X[:, 1], s=10, zorder=4)
#plt.scatter(points[:, 0], points[:, 1], s=75, c='r', zorder=3)
# Plot center.
plt.scatter(*centroid, s=70, c='g')

plt.show()