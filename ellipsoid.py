# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:57:14 2020

@author: apaudice
"""

import numpy as np

class Ellipsoid:
    """Axis-aligned d-dimensional ellipsoid.
    Attributes
    ----------
    mu : numpy.ndarray
        Center of the ellipsoid
    l : numpy.ndarray
        Length of semiaxes
    eig : numpy.ndarray
        Eigenvalues, that is, 1/l**2
    """

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
        """Rescale the ellipsoid by the specified factor about its center.
        """

        self.l*=by
        self.eig*=1.0/by**2
        
    def contains(self, x):
        """Tells whether x is contained in E.
        """

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