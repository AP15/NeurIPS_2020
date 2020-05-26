# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:02:50 2020

@author: apaudice
"""

import numpy as np
import pandas as pd
import Dataset as ds
import SCQKMeansBudget as scqkm
import ellipsoidalClusteringBudget as ecc
import matplotlib.pyplot as plt
import oracle
import datagen
    

def plotAccuracyQueries(filename):
    data = np.load(filename+'_kmeans_accuracyQueries.npz')
    scores_kmeans, queries_kmeans = data['arr_0'], data['arr_1']
    data = np.load(filename+'_ecc_accuracyQueries.npz')
    queries, scores_ecc = data['arr_2'], data['arr_3']
    queries_ecc_mean = np.nanmean(queries, axis=0)
    scores_ecc_mean = np.nanmean(scores_ecc, axis=0)
    
    print('#Queires kmeans:', int(np.mean(queries_kmeans, axis=0)))
    
    #Plot results
    f = plt.figure()
    plt.hlines(np.mean(scores_kmeans, axis=0), queries_ecc_mean[0], 
               queries_ecc_mean[-1], linestyle='--', colors='blue', label='scqkmeans')
    plt.plot(queries_ecc_mean, scores_ecc_mean, color='red', label='ecc')
    plt.fill_between(queries_ecc_mean, scores_ecc_mean - np.nanstd(scores_ecc, axis = 0)/2, 
                     scores_ecc_mean + np.nanstd(scores_ecc, axis = 0)/2, 
                     facecolor='red', alpha=0.5)
    
    plt.ylim(0, 1.1)
    
    plt.xlabel('#Queries')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_accuracyQueries.pdf", bbox_inches='tight')


def plotAccuracyBudget(filename):
    data = np.load(filename+'_kmeans_accuracy.npz')
    B, scores_kmeans = data['arr_0'], data['arr_1']
    data = np.load(filename+'_ecc_accuracy.npz')
    B, scores_ecc = data['arr_0'], data['arr_1']
    #Plot results
    f = plt.figure()
    plt.plot(B, np.mean(scores_kmeans, axis = 0), color='blue', label='kmeans')
    plt.fill_between(B, np.mean(scores_kmeans, axis = 0) - np.std(scores_kmeans, axis = 0)/2, 
                     np.mean(scores_kmeans, axis = 0) + np.std(scores_kmeans, axis = 0)/2, 
                     facecolor='blue', alpha=0.5)
    plt.plot(B, np.mean(scores_ecc, axis = 0), color='red', label='ecc')
    plt.fill_between(B, np.mean(scores_ecc, axis = 0) - np.std(scores_ecc, axis = 0)/2, 
                     np.mean(scores_ecc, axis = 0) + np.std(scores_ecc, axis = 0)/2, 
                     facecolor='red', alpha=0.5)
    
    plt.ylim(0, 1.1)
    
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.legend()
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
            s=10, 
            marker='o', 
            c=np.array(np.tile(rgb, (X_i.shape[0], 1))), 
            edgecolor='black',
            label='cluster ' + str(i+1))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_clustering.pdf", bbox_inches='tight')