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
import time

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
        Assumes x is contained in some rectangle.
        x may be a 2d array (one point per row).
        """
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

    def findPoints(self, R, X):
        """Return the indices of all points of X in rectangle R
        """
        pass

# E = Ellipsoid([0,0], [5,5])
# Ein = Ellipsoid([0,0], [1,1])
# T = Tessellation(E, Ein, .5)

class Cleaner:
    """Find points in X \cap E that belong to a cluster, by tessellating an ellipsoid with hyperrectangles.
    D is a pandas dataframe with one row per point and whose last column must be a boolean
    telling whether the point is in C; unknown labels must be np.nan.
    E is the ellipsoid used as reference for the tessellation.
    gamma is the margin.
    greedyBox tells whether to apply the greedy box growing labeling
    greedyHull tells whether to apply the greedy convex hull growing labeling
    Note that points in D are assumed to be already w.r.t. the center of the tessellation.
    """
    def __init__(self, D, E, gamma=0.1, greedy_point=True, greedy_hull=True, tess=True):
        self.gamma = gamma
        self.D = D.copy()
        self.d = self.D.shape[-1] - 1
        self.E = Ellipsoid(E.mu, E.l)
        self.Ein = Ellipsoid(E.mu, E.l/self.d)
        self.D.iloc[:,:d] = (self.D.iloc[:,:d]-E.mu).abs() # center and place in positive orthant
        # self.C = self.D[self.D.iloc[:,self.d]==True].index # labeled as C
        # self.N = self.D[self.D.iloc[:,self.d].isna()].index # yet unlabeled
        # self.U = self.D[self.D.iloc[:,self.d].isna()].index # yet unlabeled
        self.greedy_point=greedy_point
        self.greedy_hull=greedy_hull
        self.tess=tess

    def getUnlabeled(self):
        return self.D.loc[self.D.iloc[:,self.d].isna()]

    def getPositives(self):
        return self.D.loc[self.D.iloc[:,self.d]==True]

    def getNegatives(self):
        return self.D.loc[self.D.iloc[:,self.d]==False]

    def setPositive(self, idx):
        self.D.iloc[idx,self.d]=True

    def setNegative(self, idx):
        self.D.iloc[idx,self.d]=False

    def randomHullChecker(self, H, X, neq=10, feq=0.05):
        """Checks whether points in X are in the ConvexHull H by checking a random subset of inequalities.
        Return an array with entries:
        False if x is found to violate some inequality (thus certainly x not in H)
        True if no inequality has been found that is violated by x.
        neq is the max number of inequalities checked.
        feq is the max fraction of inequalities checked.
        Therefore we check min(feq*len(H.equations), neq) inequalities.
        """
        M=np.c_[X, np.ones((X.shape[0],1))] # add 1
        q=int(min(feq*H.equations.shape[0], neq))
        if q<0:
            raise Exception("You asked to check <0 inequalities.")
        Qidx=np.random.choice(H.equations.shape[0],q)
        Q=H.equations[Qidx].transpose()
        return np.all(np.dot(M,Q)<=0, axis=1)

    def findPointsInHull(self, H, X):
        print("convex hull has %d vertices, %d facets" % (len(H.vertices), len(H.equations)))
        # 1. random check to exclude most points (hopefully)
        L=self.randomHullChecker(H,X)
        print("checked %d points of which %d excluded" % (len(L), sum(L==False)))
        # 2. deterministic check on remaining points
        L[L==True]=point_in_hull(X[L],H) # actual check
        return L

    def greedyHull(self, step=None):
        """Greedily label points using the expanded convex hull.
        """
        if self.getUnlabeled().empty:
            print("greedyHull: all points are labeled, exiting.")
            return
        if step is None:
            step=self.gamma/2
        print("START greedy hull expansion with step 1+%2f" % step)
        # X=self.getPositives().iloc[:,:self.d] # to use for convex hull
        # H=ConvexHull(np.r_[(1+step)*X, [np.zeros(self.d)]])
        # active=set(H.vertices)
        orig=np.zeros(self.d)
        old=set()
        itr=nc=0
        while True:
            itr+=1
            X=self.getPositives().iloc[:,:self.d] # to use for convex hull
            H=ConvexHull(np.r_[(1+step)*X, [orig]])
            if set(H.vertices)==old:
                break
            U=self.getUnlabeled().iloc[:,:self.d] # points to test
            P=self.findPointsInHull(H,U)
            print("found %d positives " % sum(P==True))
            #isInH=point_in_hull(D.iloc[:, :d], H)
            self.setPositive(U.loc[P].index)
            old=set(H.vertices)
        print("%d iterations" % (itr,))

    def greedyPoint(self, step=None):
        """A dumb but effective labeling algorithm. Takes each labeled point x and
        sees if it dominates any unlabeled point y w.r.t. the origin, i.e., if
           y <= (1+step)x 
        where step depends on the margin between clusters. Stops when no improvement is made.
        """
        if step is None:
            step=self.gamma/2
        print("START greedy point expansion with step 1+%2f" % step)
        D, d = self.D, self.d
        U=self.getUnlabeled() # yet unlabeled points
        active=set(self.getPositives().index)
        itr=nc=0
        while active:
            itr+=1
            new_active=set()
            for idx in active:
                x=(1+step)*np.array(D.iloc[idx,:d])
                nc+=len(U)
                dx=(U.iloc[:,:d]<=x).all(axis=1) # find points dominated by x
                self.setPositive(dx[dx==True].index) # new positives
                new_active.update(dx[dx==True].index) # insert new positives
                U=U[dx==False] # remove new positives
            active=new_active.difference(active) # update active set
        print("%d iterations, %d checks" % (itr,nc))

    def tessellationClean(self):
        # 1. build tessellation
        T=Tessellation(self.E, self.Ein, self.gamma)
        R=T.findRectangle(self.D.iloc[:, :self.d])
        self.D['R']=[tuple(r) for r in R]
        self.D.sort_values(by=['label'], inplace=True) # sort by label so that NA will be last

        # 2. label each rectangle
        xref=self.getPositives()[:1]
        G=self.D.groupby('R').head(1) # group points by rectangle, pick first point
        G=G.loc[self.getUnlabeled().index] # for these points, the rectangle has no labeled point
        for idx, row in G.iterrows():
            # learn label of point
            label=(idx%2==0) # just to get a mix of True/False
            if label:
                self.setPositive([idx])
            else:
                self.setNegative([idx])
        self.D.fillna(method='ffill', inplace=True)

    def clean(self):
        self.printStatus()
        if self.greedy_point:
            self.greedyPoint(step=self.gamma/2)
            self.printStatus()
        if self.greedy_hull:
            self.greedyHull()
            self.printStatus()
        if self.tess:
            pass
            self.tessellationClean()
            self.printStatus()

    def printStatus(self):
        print("%d points, %d positives (%.2f%%)" % (len(self.D), len(self.getPositives()), 100*len(self.getPositives())/len(self.D)))

def test(n=1000, d=2, gamma=.2):
    import cProfile
    import pstats

    np.random.seed(0)
    X = np.random.normal(size=(n,d)) / np.sqrt(d)
    df = pd.DataFrame(X)
    E=Ellipsoid([0]*d, [1]*d)
    E1=Ellipsoid([0]*d, [.3]*d)
    SC=[]
    while len(SC)<0.05:
        E1=Ellipsoid([0]*d, E1.l*1.1)
        SC=df[E1.contains(X)].sample(frac=.5).index
    df['Y'] = np.nan
    df.iloc[SC,-1] = True # SC

    C=Cleaner(df, E, gamma, True)
    D=C.clean()
    print(D.shape)

    #print(D)
    #cProfile.run("C=Cleaner(df, SC, E, %2f, True)\nC.clean()" % (gamma), "clstats")
    #p=pstats.Stats("clstats")
    #p.sort_stats("cumulative").print_stats(10)

doPlot=False
n=1000
d=5
gamma=.8
sigma=1+np.arange(d)
np.random.seed(0)
X=sigma*np.random.normal(size=(n,d), scale=1)
#X=np.r_[X,sigma[::-1]*(2*d+np.random.normal(size=(n,d), scale=1))]
df=pd.DataFrame(X)
df['label'] = np.nan # create label column
SC=df[:n].sample(3*d).index # to simulate the sampled points in SC
df.iloc[SC,d] = True 
E=Ellipsoid([0]*d, l=2*np.sqrt(sigma))

C=Cleaner(df, E, gamma, greedy_point=True)
C.clean()
import matplotlib.pyplot as plt

if doPlot:
    ms=8 # marker size
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.scatter(df[0], df[1], s=ms)
    ax1.scatter(df.loc[SC,0], df.loc[SC,1], c='black', s=ms)
    ax2.scatter(df[0], df[1], s=ms)
    P=df.loc[C.getPositives().index]
    ax2.scatter(P[0], P[1], c='black', s=ms)
    plt.show(block=False)
