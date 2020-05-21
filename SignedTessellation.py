# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:51:35 2020

@author: apaudice
"""

import numpy as np
from tessellation import Tessellation

class PointOutsideTessellationError(ValueError):
    pass

class SignedTessellation(Tessellation):
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

    def __init__(self, E, Ein=None, gamma=1):
        """Construct a tessellation.

        Parameters
        ----------
        E : Ellipsoid
            The outer bounding ellipsoid
        Ein : Ellipsoid
            The inner bounding ellipsoid (ignored)
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
        x : numpy.ndarray or pandas.DataFrame
            A d-dimensional array

        Returns
        -------
        r : tuple
            A d-dimensional tuple identifying the rectangle containing x.
            For instance, (-1,2,1) identifies the rectangle formed by the
            cartesian product: (-T[0]) x T[1] x T[0]; see the T attribute.
            Note that, in order to store the sign, the ids do not include 0.

        Examples
        --------
        E=Ellipsoid(np.array([0,0]), np.array([4,4]))
        ST=st.SignedTessellation(E,gamma=.5)
        ST.findRectangle([.5,-.1])
        >> array([ 1, -1])
        """
        x = np.array(x)-self.E.mu # convert to ndarray & center
        t = np.clip(np.abs(x),.9*self.beta,np.inf) # clip to avoid log(0) just in case
        lg = np.log(t/self.beta)/np.log(1+self.a) # convert to id
        r = (np.clip(np.ceil(1+lg),1,np.inf)*np.sign(x)).astype(int) # ceiling and saturation at 1 to allow for sign
        return r

    def signedCoords(self, t, s):
        """
        Adjusts an interval according to the sign
        Parameters
        ----------
        t : np.ndarray
            A length-2 array
        s : int or float
            The sign
        Returns
        -------
        is s >= 0, just t; else, t reversed and negated
        """
        return t if s>=0 else [-t[1],-t[0]]

    def getRectangleCoords(self, rectId):
        """
        Return the coordinates of the given hyperrectangle.
        Parameters
        ----------
        rectId : np.ndarray
            A length-d array of signed integers from {-s,...,-1} U {1,...,s}
            where s=len(self.T[0]) is the number of intervals along each positive axis.

        Returns
        -------
        rt : np.ndarray
            A bidimensional array with shape (d,2), whose i-th row specifies the interval
            of the rectangle along the i-th axis, with sign.

        Examples
        --------
        E=Ellipsoid(np.array([0,0]), np.array([4,4]))
        ST=st.SignedTessellation(E,gamma=.5)
        ST.getRectangleCoords([-1,2])
        >> array([[-1.8660254 , -1.        ],
        >> [ 1.8660254 ,  3.48205081]])
        """
        if (np.abs(rectId) > len(self.T[0])).any():
            raise PointOutsideTessellationError("one of the input points is outside the tessellation range")
        return np.array([self.signedCoords(self.T[i][np.abs(rectId[i])], rectId[i]) for i in range(self.d)])
