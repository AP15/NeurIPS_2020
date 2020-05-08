#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:56:14 2020
@author: Marco Bressan
"""

import pandas as pd
import numpy as np
from scipy.spatial.qhull import ConvexHull, QhullError
import timeit
import time
import logging
import oracle
import ellipsoid as ell

# from https://stackoverflow.com/questions/16750618/
# def ppoint_in_hull(point, hull, tolerance=1e-12):
#   # VERY SLOW!
#     return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

def point_in_hull(X, H, tolerance=1e-12):
    """Check if points are in a convex hull.

    Parameters
    ----------
    X : numpy.array
        An m-by-n array of m points in R^n
    H : scipy.spatial.ConvexHull
        The convex hull
    tolerance : float
        comparison tolerance

    Returns
    -------
    Y : numpy.array
        An length-m array whose i-th entry is True iff X[i] is in H
    """

    M=np.c_[X, np.ones((X.shape[0],1))]
    return np.all(np.dot(M, H.equations.transpose()) <= tolerance, axis=1)


class Tessellation:
    """Tessellation of the space between two ellipsoids.

    Attributes
    ----------
    E : Ellipsoid
        The outer bounding ellipsoid
    Ein : Ellipsoid
        The inner bounding ellipsoid
    d : int
        Dimensionality of data
    gamma : float
        Margin of cluster
    a : float
        The rectangle increment step
    L : numpy.ndarray
        The semiaxes lengths of E
    beta : numpy.ndarray
        The boundary of the first rectangle along each axis
    b : int
        The number of intervals along each axis
    T : list
        The i-th element is the list of intervals along the i-th axis
    """

    def __init__(self, E, Ein, gamma):
        """Construct a tessellation.

        Parameters
        ----------
        E : Ellipsoid
            The outer bounding ellipsoid
        Ein : Ellipsoid
            The inner bounding ellipsoid
        gamma : float
            The margin
        """

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
        """Find the rectangle the point lies in.

        Parameters
        ----------
        x : numpy.ndarray
            A d-dimensional array

        Returns
        -------
        r : tuple
            A d-dimensional tuple identifying the rectangle containing x.
            For instance, (0,1,0) identifies the rectangle formed by the
            cartesian product: T[0] x T[1] x T[2]; see the T attribute.
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
    """Find points that belong to a cluster by tessellating with hyperrectangles.

    Attributes
    ----------
    D : pandas.DataFrame (see below)
        The input points
    E : Ellipsoid
        The region of space to be cleaned (only points in E will be labeled)
    gamma : float
        The margin of the cluster

    Methods
    -------
    getPositives()
        Return a view of the subset of points labelled as True
    getNegatives()
        Return a view of the subset of points labelled as False
    getUnlabeled()
        Return a view of the subset of unlabelled points
    setPositive(idx: pd.Index or array-like)
        Change to True the label of points with index idx
    setNegative(idx: pd.Index or array-like)
        Change to False the label of points with index idx
    printStatus()
        Prints the number of points and of positives
    greedyPoint(step: float)
        Greedy point labeling strategy based on point expansion
    greedyHull(step: float)
        Greedy point labeling strategy based on convex hull expansion
    """

    def __init__(self, D, E, gamma=0.1, d=None, logLevel=20):
        """Create a cleaner.

        Parameters
        ----------
        D : pandas.DataFrame
            The dataset holding the points. The first d columns are the point coordinates,
            and column (d+1) is the label. The label must be in {True,False,np.nan}. By
            default d is set to D.shape[1]-1 i.e. the number of columns minus one.
        E : Ellipsoid
            The Ellipsoid that bounds the region of the space to be cleaned. The center of
            E is used as center of the dataset (i.e. the margin is meant w.r.t. E.mu). The
            rescaled version of E is used as starting point for the tessellation.
        gamma : float
            The margin to be used for the tessellation and cleaners (default is 0.1).
        d : int
            Use the first d columns of D as coordinates (default is D.shape[1]-1).
        logLevel : int
            Logging level as in logging (https://docs.python.org/3/howto/logging.html)
            10=logging.DEBUG, 20=logging.INFO, 30=logging.WARN, ..(default is logging.INFO)
        """

        self.D=D.copy()
        self.d=self.D.shape[1]-1 if d is None else d
        self.E=ell.Ellipsoid(E.mu, E.l)
        self.Ein=ell.Ellipsoid(E.mu, E.l/self.d)
        self.D.iloc[:,:self.d]=self.D.iloc[:,:self.d]-E.mu # set origin at the center of E
        self.orig=np.zeros(self.d) # now set origin at zero
        self.gamma=gamma
        self.logger=logging.getLogger()
        self.logger.setLevel(logLevel)

    def getPositives(self):
        return self.D.loc[self.D.iloc[:,self.d]==True]

    def getNegatives(self):
        return self.D.loc[self.D.iloc[:,self.d]==False]

    def getUnlabeled(self):
        return self.D.loc[self.D.iloc[:,self.d].isna()]

    def setPositive(self, idx):
        self.D.loc[idx,self.D.columns[self.d]]=True

    def setNegative(self, idx):
        self.D.loc[idx,self.D.columns[self.d]]=False

    def printStatus(self):
        self.logger.info("%d points of which %d positives (%.2f%%)" % (len(self.D), len(self.getPositives()), 100*len(self.getPositives())/len(self.D)))

    def greedyPoint(self, step=None):
        """Label points using a greedy point expansion.

        Repeatedly takes the set of positives, expands it by a multiplicative factor
        (1+step), and labels as positive any point dominated by any of them. Points
        are never used twice. Stops when no more positives are found. Running time is
        O(n^2) for n points. Can be slow in practice.

        Parameters
        ----------
        step : float
            The expansion step (default depends on gamma)
        """

        if step is None:
            step=self.gamma/2
        self.logger.info("START greedy point expansion with step 1+%2f" % step)
        D, d = self.D, self.d
        D.iloc[:,:d] = (D.iloc[:,:d]-self.E.mu).abs() # center and take absolute value
        U=self.getUnlabeled() # yet unlabeled points
        active=set(self.getPositives().index)
        itr=nc=0
        while active:
            itr+=1
            new_active=set()
            for idx in active:
                x=(1+step)*np.array(D.loc[idx,range(d)])
                nc+=len(U)
                dx=(U.iloc[:,:d]<=x).all(axis=1) # find points dominated by x
                self.setPositive(dx[dx==True].index) # new positives
                new_active.update(dx[dx==True].index) # insert new positives
                U=U[dx==False] # remove new positives
            active=new_active.difference(active) # update active set
        self.logger.info("%d iterations, %d checks" % (itr,nc))

    def randomHullTest(self, H, D, neq=50, feq=0.05):
        """Exclude points from convex hull by checking a random subset of constraints.

        Parameters
        ----------
        H : scipy.spatial.ConvexHull
            The convex hull
        D : pandas.DataFrame
            The points to be tested
        neq : int
            How many random constraints to check (default is 50)
        feq : float
            The fraction of random contraints to check (default is 0.05)

        Returns
        -------
        I : pandas.Index
            The indices of the points of D that are in H
        """

        q=int(np.ceil(min(feq*H.equations.shape[0], neq)))
        self.logger.debug("randomHullTest on %d points, %d constraints" % (D.shape[0], q))
        D['ones']=1.0
        Q=H.equations[np.random.choice(H.equations.shape[0],q)]
        y=np.all(np.dot(D,Q.transpose())<=0, axis=1)
        return D.loc[y].index

    def findPointsInHull(self, H, D, useRnd=True):
        """Find which input points belong a convex hull.

        Before checking deterministically every point, optionally discards points
        by checking against a random subset of convex hull inequalities. The 
        randomized check is fast and can discard a large fraction of points. The
        final results is always deterministic.

        Parameters
        ----------
        H : scipy.spatial.ConvexHull
            The convex hull
        D : pandas.DataFrame
            The points to be tested
        useRnd : bool, optional
            Whether to use randomized exclusion test (recommended for speed)

        Returns
        -------
        I : pandas.Index
            The indices of the points of D that are in H
        """

        self.logger.debug("convex hull has %d vertices, %d facets" % (len(H.vertices), len(H.equations)))
        L=D.index
        if useRnd:
        # 1. random check to exclude most points (hopefully)
            while True:
                oldl=len(L)
                L=self.randomHullTest(H,D.loc[L])
                if len(L)>.9*oldl or len(L)==0:
                    break
        self.logger.debug("checked %d points, %d survived" % (D.shape[0], D.loc[L].shape[0]))
        # 2. deterministic check on remaining points
        Y=point_in_hull(D.loc[L].iloc[:,:self.d],H)
        self.logger.debug("found %d positives" % len(Y))
        return D.loc[Y].index

    def greedyHull(self, step=None, maxpoints=50):
        """Label points using a greedy convex hull expansion.

        Repeatedly computes the convex hull of positives, expands it by a
        multiplicative factor (1+step), and labels as positive any point
        inside it. Stops when no more positives are found.

        Parameters
        ----------
        step : float
            The expansion step (default depends on gamma).
        maxpoints : int
            If > 0, then indicates the max number of positives to use for 
            building the convex hull. Using a small number reduces the
            running time but also the labeling recall.
        """

        if self.getUnlabeled().empty:
            self.logger.debug("greedyHull: all points are labeled, exiting.")
            return
        if step is None:
            step=self.gamma/2
        self.logger.info("starting greedy hull expansion with step 1+%2f" % step)
        itr=nc=0
        while True:
            itr+=1
            X=self.getPositives().iloc[:,:self.d] # for convex hull
            if maxpoints > 0 and maxpoints < X.shape[0]:
                X=X.sample(int(maxpoints))
            self.logger.debug("building convex hull on %d points... " % len(X))
            try:
                H=ConvexHull(np.r_[(1+step)*X, [self.orig]])
            except QhullError:
                return # set of points is degenerate
            self.logger.debug("done")
            U=self.getUnlabeled().iloc[:,:self.d] # points to test
            I=self.findPointsInHull(H,U) # index of points in hull
            if len(I)==0:
                break
            self.setPositive(I)
        self.logger.debug("%d iterations" % (itr,))


    def tessellationClean(self, oc=None):
        """Classify points via hyperrectangle tessellation and oracle queries.

        Parameters
        ----------
        oc : SCQOracle
            The same-cluster-query oracle. Must provide a method scq(ix,iy) that
            tells whether points ix and iy are in the same cluster.
        """

        if self.getUnlabeled().empty:
            self.logger.debug("no unlabeled point, returning")
            return
        self.logger.info("starting tessellation")
        # 1. build tessellation
        self.T=Tessellation(self.E,self.Ein,self.gamma)
        R=self.T.findRectangle(self.D.iloc[:,:self.d].abs()) # map points to rectangles
        self.D['R']=[tuple(r) for r in R]
        self.D.sort_values(by=self.D.columns[self.d],inplace=True) # sort by label so that NA will be last
#        return
#        self.G=df
        # 2. label each rectangle
        self.G=self.D.groupby('R')
        G=self.D.groupby('R').head(1) # group points by rectangle, pick first point
        Q=G.loc[G.iloc[:,self.d].isna()] # for these points, the rectangle has no labeled point
#        print(Q)
        iC = self.getPositives().index[0] # for comparison, this point is in C
        self.logger.debug("there are %d rectangles of which %d unlabeled" % (len(G),len(Q)))
#        return
        for idx, row in Q.iterrows():
            # learn label of point
            self.logger.debug("learning label of rectangle %s from point #%d" % (Q.loc[idx,'R'],idx))
            label=oc.scq(iC, idx) # replace with Oracle call
            if label:
                self.setPositive([idx])
            else:
                self.setNegative([idx])
        self.D.fillna(method='ffill', inplace=True)
        self.logger.info("queried %d points " % len(Q))
        # restore original order, remove rectangle column
        self.D.sort_index(inplace=True)
        self.D.drop(self.D.columns[self.d+1],axis=1,inplace=True)



def test(n=1000, d=2, gamma=.2, seed=0, plot=False):
    np.random.seed(seed)
    import cProfile
    import pstats
    s=5*d # |S_C|
    sigma=1+np.arange(d)

    # uniform distribution in a ball, see
    # http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    X=np.random.normal(size=(n,d+2), scale=1)
    X=X/np.linalg.norm(X,axis=1).reshape((n,1))
    X=sigma*X[:,:d]

    E=ell.Ellipsoid([0]*d, l=sigma) # simulate E containing most points
    #X=X[E.contains(X)]

    df=pd.DataFrame(X)
    df['y'] = np.nan # start with all points unlabeled
    SC=df[E.contains(X)].sample(s).index # to simulate the sampled points in SC
    df.iloc[SC,d] = True 

    C=Cleaner(df[E.contains(X)], E, gamma)
    C.printStatus()
    C.greedyHull(maxpoints=100) # hull expansion
    C.tessellationClean() # tessellation
    C.printStatus() # see how many points left...

    #cProfile.run("C=Cleaner(df, E, %2f, greedy_point=False)\nC.clean()" % (gamma), "clstats")
    #p=pstats.Stats("clstats")
    #p.sort_stats("cumulative").print_stats(10)

    import matplotlib.pyplot as plt
    if plot:
        ms=5 # marker size
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        ax1.scatter(df[0], df[1], s=ms, c='orange')
        ax1.scatter(df.loc[SC,0], df.loc[SC,1], c='black', s=ms)
        ax2.scatter(df[0], df[1], s=ms, c='orange')
        P=df.loc[C.getPositives().index]
        ax2.scatter(P[0], P[1], c='black', s=ms)
        plt.show(block=False)

