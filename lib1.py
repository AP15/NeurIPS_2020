#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:56:14 2020

@author: brix
"""

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import timeit

# from https://stackoverflow.com/questions/16750618/
# def ppoint_in_hull(point, hull, tolerance=1e-12):
#   # VERY SLOW!
#     return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

def point_in_hull(X, H, tolerance=1e-12):
    """Checks if the points in X are in H."""
    M=np.c_[X, np.ones((X.shape[0],1))]
    return np.all(np.dot(M, H.equations.transpose()) <= tolerance, axis=1)

def point_in_hull_2(X, H, tolerance=1e-12):
    """Checks if the points in X are in H."""
    # SLOW
    return np.all(np.dot(X, H.equations[:,:-1].transpose()) + H.equations[:,-1].transpose() <= tolerance, axis=1)

class Ellipsoid:
    """An axis-aligned d-dimensional ellipsoid"""

    def __init__(self, mu, l=None, eig=None):
        self.mu=np.array(mu, dtype=float)
        self.d = len(self.mu)
        if l is None and eig is None:
            raise Exception("One between semiaxes length and eigenvalues must be specified.")
        if l is not None:
            self.l = np.array(l, dtype=float)
            self.eig = 1.0/self.l**2
        if eig is not None:
            self.eig = np.array(eig, dtype=float)
            self.l = 1.0/np.sqrt(self.eig)
    
    def __str__(self):
        return "center: " + np.array_str(self.mu) + "\nsemiaxes: " + np.array_str(self.l) + "\neigenvalues: " + np.array_str(self.eig)

    def rescale(self, by=1):
        """Rescale the ellipsoid by the specified factor about its center."""
        self.l*=by
        self.eig*=1.0/by**2
        
    def contains(self, x):
        """Tells whether x is contained in E."""
        return np.dot((np.array(x)-self.mu)**2,self.eig)<= 1

    def move_to(self, new_mu):
        self.mu = np.array(new_mu)

#def testEllipsoid():
#    E = Ellipsoid([0,0], [1,1])
#    E.contains([1,1])
#    E.contains([1/2,-1/2])
#    E = Ellipsoid([0,0], eig=[4,4])
#    E.contains([-1/2,0])
#    E.contains([-1/2,0.1])


class Tessellation:
    """Tessellation of the space between two ellipsoids."""

    def __init__(self, E, Ein, gamma):
        self.E = E
        self.Ein = Ein
        self.d = self.E.mu.shape[-1]
        self.gamma = gamma
        self.a = np.sqrt((1+self.gamma)/2)
        self.L = self.E.l
        self.beta = self.L * np.sqrt(self.gamma/2)/self.d
        self.b = int(np.ceil(np.log(self.L[0]/self.beta[0]) / np.log(1+self.a)))
        self.T = []
        for i in range(self.d):
            # for each axis build the list of intervals, t
            t = [np.array([0, self.beta[i]])]
            t = t + [self.beta[i]*np.array([(1+self.a)**j, (1+self.a)**(j+1)]) for j in range(self.b-1)]
            self.T.append(t)
    
    def findRectangle(self, x):
        """Return a tuple identifying the rectangle containing x.
        Assumes x is contained in some rectangle."""
        # center; map in the positive orthant; clip to avoid log(0) below
        x = x.copy()
        x = np.abs(np.array(x)-self.E.mu.clip(min = self.beta/2))
        lg = np.log(x/self.beta)/np.log(1+self.a) # convert to id
        r = np.floor(lg).clip(min=0).astype(int) # ceiling and saturation at 0
        return r

    def getRectangleCoords(self, rectId):
        """Return the intervals for the given rectangle
        """
        return [self.T_[i][rectId[i]] for i in range(self.d_)]

    def classify(self, X):
        """Map points of X to their rectangles
        """
        X = X.copy()
        X = np.abs(np.array(X) - self.E.mu)
        for i in range(self.d):
            X[:,i] = X[:,i].clip(self.beta[i]/2)
        lg = np.log(X/self.beta)/np.log(1+self.a)
        I = np.floor(lg).clip(min=0).astype(int)
        return I

    def findPoints(self, R, X):
        """Return the indices of all points of X in rectangle R
        """
        pass

# E = Ellipsoid([0,0], [5,5])
# Ein = Ellipsoid([0,0], [1,1])
# T = Tessellation(E, Ein, .5)

class Cleaner:
    """Return all points in X \cap E that belong to cluster C.
    X is the whole dataset.
    SC is a list of indices corresponding to points of X that are known to be in C.
    E is an ellipsoid that contains SC.
    gamma is the margin.
    """
    def __init__(self, X, SC, E, gamma=0.1, X_has_labels=False):
        self.gamma = gamma
        self.D = pd.DataFrame(X)
        self.d = self.D.shape[-1] - int(X_has_labels)
        if not X_has_labels:
            self.D['Y'] = np.nan
        self.E = E
        self.Ein = Ellipsoid(E.mu, E.l/self.d)
        self.SC = SC

    def clean(self):
        D, d = self.D, self.d

        # 1. discard points not in E
        D = D[self.E.contains(D.iloc[:, :d])]

        # 2. label as True points in conv(SC)
        if len(self.SC) > d+1:
            H = ConvexHull(D.loc[self.SC].iloc[:, :d])
            self.H = H
            isInH = point_in_hull(D.iloc[:, :d], H)
            D.at[isInH,'Y'] = True

        # 3. tessellation
        T = Tessellation(self.E, self.Ein, self.gamma)
        R = T.classify(D.iloc[:, :d])
        D['R'] = [tuple(r) for r in R]
        D.sort_values(by=['R', 'Y'], inplace=True)

        # 4. cleaning
        x = self.SC[0]
        G = D.groupby('R').head(1)
        G = G[G.Y.isna()]
        for idx, row in G.iterrows():
            G.at[idx,'Y'] = (idx%2==0) # just to get a mix of True/False
            #G.at[idx, 'Y'] = oracle.scq(x, idx)
        D.loc[G.index,'Y'] = G['Y']

        D.fillna(method='ffill', inplace=True)
        self.D = D
        return D[D.Y == True]

    def greedyClean(self):
        pass


def test():
    import cProfile
    import pstats

    n, d, gamma = 1000, 10, 0.2
    np.random.seed(0)
    X = np.random.normal(size=(n,d)) / np.sqrt(d)
    df = pd.DataFrame(X)
    E=Ellipsoid([0]*d, [1]*d)
    E1=Ellipsoid([0]*d, [.3]*d)
    SC=df[E1.contains(X)].sample(frac=.5).index
    df['Y'] = np.nan
    df.iloc[SC,-1] = True # SC

    # C=Cleaner(df, SC, E, 0.2, True)
    # D=C.clean()
    #print(D)
    cProfile.run("C=Cleaner(df, SC, E, %2f, True)\nC.clean()" % (gamma), "clstats")
    p=pstats.Stats("clstats")
    p.sort_stats("cumulative").print_stats(10)
