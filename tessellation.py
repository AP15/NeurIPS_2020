# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:51:35 2020

@author: apaudice
"""

import numpy as np


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