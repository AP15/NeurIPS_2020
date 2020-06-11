# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:16:49 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import Dataset as dataset
import SCQKMeansBudget as scqkm
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
    expQueriesDimensions(): Run an experiment that measure the number of 
    queries asked by SCQKMeans and ECC to cluster a given dataset as a function
    of the number of dimensions.
    dataGeneration(): Generate a datasets with ellipsoidal clusters satifyng 
    the given constraints.
    """

    def __init__(self):
        pass
    
    def expQueriesDimensions(self, rep = 5):
        
        #Experiment parameters
        n_exp = 4
        n = 100000
        k = 5
        g = 1
        c = 100
        d = np.linspace(2, 8, n_exp)
        shape = 'general'
        
        #Initialization
        scoresKM = np.zeros((rep, n_exp))
        queriesKM = np.zeros((rep, n_exp))
        scoresECC = np.zeros((rep, n_exp))
        queriesECC = np.zeros((rep, n_exp))
        algKM = scqkm.SCQKmeans(k)
        algECC = ecc.ECC(k, 10)
        
        for j in range(n_exp):
            X_data, y_data, n, k = self.dataGeneration(shape, n, int(d[j]), k, 
                                                       gamma=g, rank=int(d[j]), 
                                                       cn=c)
            for i in range(rep):
                print('**************************************************************')
                print('Rep:', str(i+1) + '/' + str(rep) + '. Dim: ' 
                      + str(d[j]) + '.')
                print('SCQ-KMeans')
                X = pd.DataFrame(X_data)
                O = oracle.SCQOracle(pd.DataFrame(y_data))
                y_pred, queriesKM[i, j], temp_s, temp_q = algKM.clusterMonitor(X, O, y_data)
                scoresKM[i, j] = temp_s[-1]
                print('ECC')
                X = pd.DataFrame(X_data)
                O = oracle.SCQOracle(pd.DataFrame(y_data))
                y_pred, queriesECC[i, j], temp_s, temp_q = algECC.clusterMonitor(X, O, y_data)
                scoresECC[i, j] = temp_s[-1]
        
        np.savez('ExpDim', scoresKM, queriesKM, scoresECC, queriesECC)
        

    def expAccuracyQueries(self, dataset, algorithm='kmeans', X_data = None, 
                          y_data = None, n=1000, d=2, k=3, gamma=1, 
                          rep = 5,  showClustering=False):
        
        g = gamma
        
        scores = np.zeros(rep)
        queries = np.zeros(rep)
        queries_round, scores_round = [], []
        
        if (algorithm=='kmeans'):
            alg = scqkm.SCQKmeans(k)
        elif(algorithm=='white_kmeans'):
            alg = scqkm.SCQKmeans(k)
        elif (algorithm=='ecc'):
            alg = ecc.ECC(k, g+10)
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
            #print('Errors:', sum(y_pred!=y_data))
            #print('Unclassified:', sum(np.isnan(y_pred)))
            #print('Predictions:', y_pred)
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
        
        np.savez(dataset + '_' + str(d) + '_' + algorithm + '_accuracyQueries', 
                 scores, queries, queries_matrix, scores_matrix)
        
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
# =============================================================================
#         elif (data=='parallel'):
#             ds =  dataset.Dataset(n, d, k)
#             ds.generateEllipsoids()
#             ut.plotClustering(ds.X_, k, ds.y_, 'parallel')
#             X_data = ds.X_
#             y_data = ds.y_
# =============================================================================
        elif(data=='parallel'):
            X_, y_, Ws, cs = datagen.randomDatasetParallel(n=n, k=k, d=d, 
                                                   gamma=gamma, r = rank, 
                                                   cn=cn, tightMargin=True)
            #ut.plotClustering(X_, k, y_, 'moons')
            X_data = X_
            y_data = y_
        elif (data=='general'):
            X_, y_, Ws, cs = datagen.randomDataset(n=n, k=k, d=d, 
                                                   gamma=gamma, r = rank, 
                                                   cn=cn, tightMargin=True)
            #ut.plotClustering(X_, k, y_, 'general')
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