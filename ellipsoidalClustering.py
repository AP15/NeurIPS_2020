# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:11:03 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import scipy.linalg
import qinfer.utils as qut
import ellipsoid
import cleaning

class ECC(object):
        """
        Find ellipsoidal clusters using SCQs.
        
        Attributes
        ----------
        k : int
            #Clusters to find.
        gamma : float
            Margin parameter.
        B : int
            Query budget.
                       
        Methods
        -------
        __init__()
        sample()
        cluster()
        
        """
        
        
        def __init__(self, k, gamma):
            self.k = k
            self.gamma = gamma
            

        def conditionMVE(self, M):
            tol = 0
            eigVals, eigVecs = np.linalg.eig(M)
            #print('MVE: ', M)
            print('Eigenvalues: ', eigVals)
            eigVals[eigVals < 0] = 0
            eigVals = np.diag(eigVals)
            eigVecs_inv = np.linalg.inv(eigVecs)
            return np.linalg.multi_dot([eigVecs, eigVals, eigVecs_inv])


        def sampleT(self, population_size, m):
            """Sample u.a.r. from a population with replacement. 
            Return the samples.
            
            Parameters
            ----------
            population_size : int
                Population size.
            m : int
                Sample size.
            
            Return
            ------
            numpy array, shape = [#points]
        
            """
            
            idxs = np.random.choice(population_size, m)
            
            return idxs
        
        
        def findSeparator(self, X, S_C, S, data):
            # 3. Compute the separator, change coordinate system
            d = len(X.columns)
            M_mve, mu_mve = qut.mvee(X.values) # MVEE
            R_mve = scipy.linalg.sqrtm(M_mve) # Rotation/Stretch matrix
            Xt = data.copy() 
            # project dataset on the eigenvectors of the MVE
            Xt.iloc[:,:-1] = np.dot(Xt.iloc[:,:-1] - mu_mve, R_mve)
            # Find the separator which is the unit ball, with some tolerance
            E = ellipsoid.Ellipsoid(np.zeros(d),l=1.1*np.ones(d)) 
            # the subset we're going to clean
            D = Xt[E.contains(Xt.iloc[:,:-1])] 
            D.loc[S_C, 'y'] = True # in C
            nSC = D.index.intersection(S.difference(S_C))
            if not nSC.empty:
                D.loc[nSC, 'y'] = False # not in C
            return D, E    
        
        
        def findSeparatorC(self, X, S_C, S, data):
            # 3. Compute the separator, change coordinate system
            d = len(X.columns)
            M_mve, mu_mve = qut.mvee(X.values) # MVEE
            #Condition
            M_mve_c = self.conditionMVE(M_mve)
            R_mve = scipy.linalg.sqrtm(M_mve_c) # Rotation/Stretch matrix
            if R_mve.dtype != np.float:
                print('ERROR!!!!!!!!!!!!!!!!')
                print(R_mve)
            Xt = data.copy() 
            # project dataset on the eigenvectors of the MVE
            Xt.iloc[:,:-1] = np.dot(Xt.iloc[:,:-1] - mu_mve, R_mve)
            # Find the separator which is the unit ball, with some tolerance
            E = ellipsoid.Ellipsoid(np.zeros(d),l=1.1*np.ones(d)) 
            # the subset we're going to clean
            D = Xt[E.contains(Xt.iloc[:,:-1])] 
            print(S_C)
            print(D)
            D.loc[S_C, 'y'] = True # in C
            nSC = D.index.intersection(S.difference(S_C))
            if not nSC.empty:
                D.loc[nSC, 'y'] = False # not in C
            return D, E 

        
        def removeFalsePositives(self, X, E, oracle):
            print("removing false positives...")
            cleaner = cleaning.Cleaner(X, E, self.gamma)
            cleaner.greedyHull()
            cleaner.tessellationClean(oracle)
            #print("%d points labeled as %d" 
            #      %(cleaner.getPositives().shape[0], C))  
            return cleaner.getPositives().index               
 

        def cluster(self, data, oracle):
            #Get active nodes      
            activeNodes = data.copy()
            n = activeNodes.shape[0]  
            C = np.nan*np.ones(n)
            while (n > 0):
                print('*********************************')
                print("%d points left to cluster." % n)
                # 1. Sample
                print("1. Taking samples")
                B = min(n, 10 * self.k)
                S = activeNodes.sample(B).index
                S_labels = np.array([oracle.label(i) for i in S]) 
                if B==n: # we have labeled all of X
                    C[S]=S_labels
                    break
                C[S] = S_labels
                
                nc = pd.DataFrame({'idx': S, 'lab': S_labels}).groupby('lab').count() # number of samples per cluster
                p=nc.sort_values(by='idx',ascending=False).index[0] # id of the largest cluster sample
                S_C = S[S_labels==p] #Idx of largest cluster
                X_SC = data.loc[S_C].iloc[:,:-1] #Datapoints of the largest cluster
                print("Got %d points from cluster %d" % (len(S_C), p))
                
                # 2. Compute the separator, change coordinate system
                print("2. Computing separator between %d and %d points." 
                      % (len(S_C), len(S)-len(S_C)))
                D, E = self.findSeparatorC(X_SC, S_C, S, data)
                
                # 3. Remove false positives
                print("3. Removing false positives...")
                pos_idxs = self.removeFalsePositives(D, E, oracle)
            
                # Update labels
                C[pos_idxs] = p
                activeNodes = data.loc[np.isnan(C)]
                n = activeNodes.shape[0]
                
            return C, oracle.getCount()               
