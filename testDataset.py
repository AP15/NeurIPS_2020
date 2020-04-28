# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:52:47 2020

@author: apaudice
"""

import Dataset as ds

# =============================================================================
#Test: importFromFile
# data = Dataset()
# data.importFromFile('aggregation.txt')
# data.scatterData()
# =============================================================================

 
# =============================================================================
# #Test: generateEllipsoid
# data = Dataset(n=10000, d=2, k=3)
# data.generateEllipsoids()
# data.scatterData()
# =============================================================================


#Test: generateEllipsoidGeneral
data = ds.Dataset(n=10000, d=2, k=3)
data.generateEllipsoidsGeneral()
data.scatterData()


# =============================================================================
# #Test: sample
# data = Dataset(n=10, d=2)
# data.generate()
# 
# print('Test sample.')
# print('------------\n')
# 
# Z, y = data.sample(5)
# 
# print('Samples.\n', Z)
# print('Labels:', y, '\n')
# 
# print('Mean computation.')
# print('------------\n')
# 
# unique, counts = np.unique(y, return_counts=True)
# 
# print('Sampled clusters: ', unique) 
# print('#points per cluster: ', counts)
# 
# p = np.argmax(counts)
# 
# print('Largest sampled cluster:', p)
# print(Z[np.where(y == p)])
# print('Mean:', np.mean(Z[np.where(y == p)], axis = 0), '\n')
# 
# #Test: removePoints
# #Test: sample
# print('Test removePoints.')
# print('------------\n')
# 
# print('Data:')
# print(data.X_, '\n')
# 
# idx = np.array([0, 1])
# data.removePoints(idx)
# 
# print('New data:')
# print(data.X_)
# =============================================================================

# =============================================================================
# #Find clusering
# km = KMeans(n_clusters=3, 
#             init='random',
#             n_init=10,
#             max_iter=300,
#             tol=1e-04,
#             random_state=0)
# 
# y_km = km.fit_predict(data.X_)
# 
# plt.figure()
# plt.scatter(data.X_[y_km == 0, 0], data.X_[y_km == 0, 1],
#             s=50, 
#             c='lightgreen', 
#             marker='s', 
#             edgecolor='black',
#             label='cluster 1')
# 
# plt.scatter(data.X_[y_km == 1, 0], data.X_[y_km == 1, 1], 
#             s=50, 
#             c='orange', 
#             marker='o', 
#             edgecolor='black',
#             label='cluster 2')
# 
# plt.scatter(data.X_[y_km == 2, 0], data.X_[y_km == 2, 1],
#             s=50, 
#             c='lightblue',
#             marker='v', 
#             edgecolor='black',
#             label='cluster 3')
# 
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
#             s=250, 
#             marker='*',
#             c='red', 
#             edgecolor='black',
#             label='centroids')
# 
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()
# =============================================================================