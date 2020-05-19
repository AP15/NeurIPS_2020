# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:02:50 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import Dataset as ds
import SCQKmeansBudget as scqkm
import matplotlib.pyplot as plt
import oracle
import datagen


def plotAccuracyBudget(filename):
    data = np.load(filename+'_accuracy.npz')
    B, scores = data['arr_0'], data['arr_1']
    #Plot results
    f = plt.figure()
    plt.plot(B, np.mean(scores, axis = 0))
    #plt.errorbar(B*data.k**2, np.mean(scores, axis = 0), np.std(scores, axis = 0))
    plt.fill_between(B, np.mean(scores, axis = 0) - np.std(scores, axis = 0)/2, 
                     np.mean(scores, axis = 0) + np.std(scores, axis = 0)/2, 
                     facecolor='blue', alpha=0.5)
    
    plt.ylim(0, 1.1)
    
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_accuracyBudget.pdf", bbox_inches='tight')


def plotClustering(X, k, C, filename):
    f = plt.figure()
    k = np.unique(C)
    for i in k:
        r = np.random.uniform(0,1)
        g = np.random.uniform(0,1)
        b = np.random.uniform(0,1)
        rgb = np.array([r, g, b])
        
        X_i = X[C==i, :]
        
        plt.scatter(X_i[:, 0], X_i[:, 1],
            s=50, 
            marker='s', 
            c=np.array(np.tile(rgb, (X_i.shape[0], 1))), 
            edgecolor='black',
            label='cluster ' + str(i+1))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_clustering.pdf", bbox_inches='tight')
    
    
def experiment1(dataname, n=0, d=0, k=0, rep = 5, showClustering=False):
    n_exp = 10
    
    scores = np.zeros((rep, n_exp))
    queries = np.zeros((rep, n_exp))
    
    if (dataname=='aggregation'):
        #aggregation
        data =  ds.Dataset()
        data.importFromFile('aggregation.txt')
        npoints = data.n
        nk = 7
        plotClustering(data.X_, nk, data.y_, 'aggregation')
        X_data = data.X_
        y_data = data.y_
    elif (dataname=='isotropic'):
        npoints = n
        nk = k
        dim = d
        data =  ds.Dataset(n=npoints, d=dim, k=nk)
        data.generate()
        plotClustering(data.X_, nk, data.y_, 'isotropic')
        X_data = data.X_
        y_data = data.y_
    elif (dataname=='parallel'):
        npoints = n
        nk = k
        dim = d
        data =  ds.Dataset(n=npoints, d=dim, k=nk)
        data.generateEllipsoids()
        plotClustering(data.X_, nk, data.y_, 'parallel')
        X_data = data.X_
        y_data = data.y_
    elif (dataname=='general'):
        npoints = n
        nk = k
        dim = d
        gamma = .5
        rank = dim
        np.random.seed(0)
        n,d,k=100000,2,10
        data = ds.Dataset(n,d,k)
        data.generateEllipsoidsGeneral()
        data.randomRotateAndStretch()
        #X_, y_, Ws, cs = datagen.randomDataset(n=npoints, k=nk, d=dim, r=rank,
        #                                       gamma=gamma, tightMargin=True)
        plotClustering(data.X_, nk, data.y_, 'general')
        X_data = data.X_
        y_data = data.y_
        #X_data = X_
        #y_data = y_
    
    B = np.linspace(10*np.log(npoints), npoints*nk*0.2, n_exp)
    alg = scqkm.SCQKmeans(nk)

    for i in range(rep):
        for j in range(n_exp):
            X = pd.DataFrame(X_data)
            O = oracle.SCQOracle(pd.DataFrame(y_data))
            print('**************************************************************')
            print('Rep:', str(i+1) + '/' + str(rep) + '.Exp:', str(j+1) + '/' + str(n_exp) +'.')
            y_pred, queries[i, j] = alg.cluster(X, O, int(B[j]))
            scores[i, j] = sum(y_pred==y_data)/npoints
            if (j == n_exp-1 and showClustering):
                showClustering(X.values, nk, y_pred, 'Aggregation')
    
    
    np.savez(dataname + '_accuracy', B, scores)