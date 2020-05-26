# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:16:49 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import Dataset as dataset
import SCQKMeansBudget as scqkm
import ellipsoidalClusteringBudget as eccOld
import ecc as ecc
import matplotlib.pyplot as plt
import oracle
import datagen
import utility as ut

class Experiments(object):
    """Utilities for experiments.
    
    Methods
    ------------
    __init__(): Empty constructor.
    expAccuracyBudget(): Run an experiment on a given class of data for 
    measuring the achieved accuracy for a given SCQ budget. Compare
    SCQ k-means with ECC.
    expAccuracyQueries(): Run an experiment on a given class of data to 
    measure the accuracy as a function of the queries done along rhe rounds. 
    Compare SCQ k-means with ECC.
    dataGeneration(): Generate a datasets with ellipsoidal clusters satifyng 
    the given constraints.
    """

    def __init__(self):
        pass

    def expAccuracyQueries(self, dataset, algorithm='kmeans', X_data = None, 
                          y_data = None, n=1000, d=2, k=3, gamma=1, 
                          rep = 5,  showClustering=False):
        
        g = gamma
        
        scores = np.zeros(rep)
        queries = np.zeros(rep)
        queries_round, scores_round = [], []
        
        if (algorithm=='kmeans'):
            alg = scqkm.SCQKmeans(k)
        elif (algorithm=='eccOld'):
            alg = eccOld.ECC(k, g)
        elif (algorithm=='ecc'):
            alg = ecc.ECC(k, g)
        else:
            print('Invalid algorithm')
            return
        
        max_len = 0
        for i in range(rep):
            X = pd.DataFrame(X_data)
            O = oracle.SCQOracle(pd.DataFrame(y_data))
            print('**************************************************************')
            print('Rep:', str(i+1) + '/' + str(rep))
            y_pred, queries[i], temp_s, temp_q = alg.clusterMonitor(X, O, y_data)
            scores[i] = sum(y_pred==y_data)/n
            scores_round.append(temp_s)
            queries_round.append(temp_q)
            if (len(temp_q) > max_len):
                max_len = len(temp_q)
            #ut.plotClustering(X_data, k, y_pred, 'clustering')
        
        #Average #queries
        queries_matrix = np.nan*np.ones((rep, max_len)) 
        scores_matrix = np.nan*np.ones((rep, max_len))
        for i in range(rep):
            fill = max_len - queries_round[i].size
            queries_matrix[i, :queries_round[i].size] = queries_round[i]
            scores_matrix[i, :queries_round[i].size] = scores_round[i]
        
        np.savez(dataset + '_' + algorithm + '_accuracyQueries', scores, queries, queries_matrix, scores_matrix)
        
    
    def expAccuracyBudget(self, dataset, algorithm='kmeans', X_data = None, 
                          y_data = None, n=1000, d=2, k=3, gamma=1, 
                          n_exp=5, rep = 5, 
                          showClustering=False):
        
        g = gamma
        
        scores = np.zeros((rep, n_exp))
        queries = np.zeros((rep, n_exp))
        
        B = np.linspace(10*np.log(n), 50*np.sqrt(n), n_exp)
        
        if (algorithm=='kmeans'):
            alg = scqkm.SCQKmeans(k)
        elif (algorithm=='eccOld'):
            alg = eccOld.ECC(k, g)
        elif (algorithm=='ecc'):
            alg = ecc.ECC(k, g)
        else:
            print('Invalid algorithm')
            return
    
        for i in range(rep):
            for j in range(n_exp):
                X = pd.DataFrame(X_data)
                O = oracle.SCQOracle(pd.DataFrame(y_data))
                print('**************************************************************')
                print('Rep:', str(i+1) + '/' + str(rep) + '.Exp:', str(j+1) + '/' + str(n_exp) +'.')
                y_pred, queries[i, j] = alg.cluster(X, O, int(B[j]))
                scores[i, j] = sum(y_pred==y_data)/n
                if (j == n_exp-1 and showClustering):
                    showClustering(X.values, k, y_pred, dataset)
        
        np.savez(dataset + '_' + algorithm + '_accuracy', B, scores, queries)
        
        
    def dataGeneration(self, data, n, d, k, gamma = 1, rank = None, cn = 1):
        #np.random.seed(0)
        if (data=='aggregation'):
            ds =  dataset.Dataset()
            ds.importFromFile('aggregation.txt')
            ut.plotClustering(ds.X_, 7, ds.y_, 'aggregation')
            n = dataset.n
            k = 7
            X_data = ds.X_
            y_data = ds.y_
        elif (data=='isotropic'):
            X_, y_, Ws, cs = datagen.randomDataset(n=n, k=k, d=d, 
                                                   gamma=gamma, r = rank, 
                                                   cn=1, tightMargin=True)
            ut.plotClustering(X_, k, y_, 'general')
            X_data = X_
            y_data = y_
        elif (data=='parallel'):
            ds =  dataset.Dataset(n, d, k)
            ds.generateEllipsoids()
            ut.plotClustering(ds.X_, k, ds.y_, 'parallel')
            X_data = ds.X_
            y_data = ds.y_
        elif (data=='general'):
            X_, y_, Ws, cs = datagen.randomDataset(n=n, k=k, d=d, 
                                                   gamma=gamma, r = rank, 
                                                   cn=cn, tightMargin=True)
            ut.plotClustering(X_, k, y_, 'general')
            X_data = X_
            y_data = y_
        elif (data=='moons'):
            X_, y_, Ws, cs = datagen.randomDatasetMoons(n=n, k=k, d=d, 
                                                   gamma=gamma, r = rank, 
                                                   cn=cn, tightMargin=True)
            ut.plotClustering(X_, k, y_, 'moons')
            X_data = X_
            y_data = y_
        else:
            print('Invalid Dataset!')
            X_data = None
            y_data = None
            
        return X_data, y_data, n, k