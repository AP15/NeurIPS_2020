#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:56:14 2020

@author: brix
"""

import numpy as np

class Ellipsoid:
    
    def __init__(self, mu, l = None, eig = None):
        self.mu_ = np.array(mu, dtype=float)
        if l is None and eig is None:
            raise Exception("One between semiaxes length and eigenvalues must be specified.")
        if l is not None:
            self.l_ = np.array(l, dtype=float)
            self.eig_ = 1.0/self.l_**2
        if eig is not None:
            self.eig_ = np.array(eig, dtype=float)
            self.l_ = 1.0/np.sqrt(self.eig_)
    
    def mu(self):
        return self.mu_
    
    def l(self):
        return self.l_
    
    def eig(self):
        return self.eig_
    
    def rescale(self, by=1):
        """Rescale the ellipsoid by the specified factor about its center."""
        self.l_ *= by
        self.eig_ *= 1.0/by**2
        
    def contains(self, x):
        x = (np.array(x)-self.mu_)**2
        return np.dot(x, self.eig_) <= 1

    def move_to(self, new_mu):
        self.mu_ = new_mu

#def testEllipsoid():
#    E = Ellipsoid([0,0], [1,1])
#    E.contains([1,1])
#    E.contains([1/2,-1/2])
#    E = Ellipsoid([0,0], eig=[4,4])
#    E.contains([-1/2,0])
#    E.contains([-1/2,0.1])
