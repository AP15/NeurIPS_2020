# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:40:37 2020

@author: apaudice
"""

import numpy as np
import Dataset as ds
import SCQKmeans_np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import copy

n_exp = 10
rep = 50

scores = np.zeros((rep, n_exp))

alg = SCQKmeans_np.SCQKmeansNP()

# =============================================================================
# #aggregation
# data =  ds.Dataset()
# data.importFromFile('aggregation.txt')
# data.scatterData()
# B = np.linspace(1, np.log(data.n), n_exp)
# 
# for i in range(rep):
#     for j in range(n_exp):
#         data_p = copy.deepcopy(data)
#         print('Rep:', str(i+1) + '/' + str(rep) + '.Exp:', str(j+1) + '/' + str(n_exp) +'.')
#         y_pred = alg.cluster(data_p, data_p.k, int(B[j]))
#         scores[i, j] = accuracy_score(data.y_, y_pred)
# =============================================================================

#ellipsoid
data =  ds.Dataset(n=10000, d=100, k=10)
#data.generateEllipsoids()
data.generateEllipsoidsGeneral()
#data.generate()
#data.scatterData()

B = np.linspace(1, np.log(data.n), n_exp)

for i in range(rep):
    for j in range(n_exp):
        data_p = copy.deepcopy(data)
        print('Rep:', str(i+1) + '/' + str(rep) + '.Exp:', str(j+1) + '/' + str(n_exp) +'.')
        y_pred = alg.cluster(data_p, data_p.k, int(B[j]))
        scores[i, j] = accuracy_score(data.y_, y_pred)
        scores[i, j] += y_pred[y_pred==-1].size/data.n


#Plot results
plt.figure()
plt.plot(B*data.k**2, np.mean(scores, axis = 0))
#plt.errorbar(B*data.k**2, np.mean(scores, axis = 0), np.std(scores, axis = 0))
plt.fill_between(B*data.k**2, np.mean(scores, axis = 0) - np.std(scores, axis = 0)/2, 
                 np.mean(scores, axis = 0) + np.std(scores, axis = 0)/2, 
                 facecolor='blue', alpha=0.5)

plt.ylim(0, 1.2)

plt.xlabel('Queries')
plt.ylabel('Accuracy')
plt.grid()
plt.show()