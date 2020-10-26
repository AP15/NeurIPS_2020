import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib

def pcaData(X_train, n_pca):
    stdsc = StandardScaler()
    pca = decomposition.PCA(n_components=2)
    
    X_std = stdsc.fit_transform(X_train)
    X_pca = pca.fit_transform(X_std)
    
    return X_pca

def plotQueriesDimensions(filename):    
    #data = np.load('ExpDim.npz')
    data = np.load('ExpDim.npz')
    scores_kmeans, queries_kmeans = data['arr_0'], data['arr_1']
    scores_ecc, queries_ecc = data['arr_2'], data['arr_3']

    d = np.linspace(2, 8, 4)    
    queries_kmeans_mean = np.mean(queries_kmeans, axis=0)
    scores_kmeans_mean = np.mean(scores_kmeans, axis=0)
    queries_ecc_mean = np.mean(queries_ecc, axis=0)
    scores_ecc_mean = np.mean(scores_ecc, axis=0)
    
    print('#Queries ECC:', queries_ecc_mean)
    print('#Queries KMeans:', queries_kmeans_mean)
    
    #Plot results
    f = plt.figure()
    plt.plot(d, queries_ecc_mean, color='red', label='ecc')
    plt.fill_between(d, queries_ecc_mean - np.std(queries_ecc, axis = 0)/2, 
                     queries_ecc_mean + np.std(queries_ecc, axis = 0)/2, 
                     facecolor='red', alpha=0.5)
    plt.plot(d, queries_kmeans_mean, color='blue', label='scq-k-means')
    plt.fill_between(d, queries_kmeans_mean - np.std(queries_kmeans, axis = 0)/2, 
                     queries_kmeans_mean + np.std(queries_kmeans, axis = 0)/2, 
                     facecolor='blue', alpha=0.5)
    
    #plt.ylim(0, 1.1)
    
    plt.xlabel('d')
    plt.ylabel('#Queries')
    plt.legend()
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_QueriesDimensions.eps", bbox_inches='tight')

def plotAccuracyQueriesPCA2(filename, d):
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    data = np.load("Data/"+filename+"_"+str(d)+'_kmeans_accuracyQueries.npz')
    scores_kmeans, queries_kmeans = data['arr_0'], data['arr_1']
    data = np.load("Data/"+filename+"_"+str(d)+'_white_kmeans_accuracyQueries.npz')
    scores_w_kmeans, queries_w_kmeans = data['arr_0'], data['arr_1']
    data = np.load("Data/"+filename+"_"+str(d)+'_ecc_accuracyQueries.npz')
    queries, scores_ecc = data['arr_2'], data['arr_3']
    queries_ecc_mean = np.nanmean(queries, axis=0)
    scores_ecc_mean = np.nanmean(scores_ecc, axis=0)
    
    avg_queries_kmeans = np.mean(queries_kmeans, axis=0)
    avg_queries_w_kmeans = np.mean(queries_w_kmeans, axis=0)
    print('#Queries kmeans:', int(np.mean(queries_kmeans, axis=0)))
    print('#Queries ecc:', int(queries_ecc_mean[-1]))
    print('Accuracy kmeans:', np.mean(scores_kmeans, axis=0))
    print('Accuracy ecc:', scores_ecc_mean[-1])    
    
    #Plot results
    f = plt.figure(figsize=(7,6))
    plt.hlines(1-np.mean(scores_kmeans, axis=0), avg_queries_kmeans, 
               queries_ecc_mean[-1], linestyle='--', colors='blue', 
               label='SCQ-k-means')
    
    plt.hlines(1-np.mean(scores_w_kmeans, axis=0), avg_queries_w_kmeans, 
                queries_ecc_mean[-1], linestyle=':', colors='green', label='whitened-SCQ-k-means')
    
    plt.scatter(avg_queries_kmeans, 1-np.mean(scores_kmeans, axis=0),
                          color='blue', marker='o')
    plt.scatter(avg_queries_w_kmeans, 1-np.mean(scores_w_kmeans, axis=0),
                      color='green', marker='o')
    plt.plot(queries_ecc_mean, 1-scores_ecc_mean, 
             marker='o', color='red', label='RECUR')
# =============================================================================
#     plt.fill_between(queries_ecc_mean, scores_ecc_mean - np.nanstd(scores_ecc, axis = 0)/2, 
#                      scores_ecc_mean + np.nanstd(scores_ecc, axis = 0)/2, 
#                      facecolor='red', alpha=0.5)
# =============================================================================
    
    plt.ylim(0, 1)
    plt.xlim(0, queries_ecc_mean[-1])
    
    ft = 20
    plt.xlabel('queries', fontsize=ft)
    plt.ylabel('clustering error',  fontsize=ft)
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.legend(fontsize=ft, framealpha=1)
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_" + str(d) + "_" + "PCA_errorQueries.eps", bbox_inches='tight')

def plotAccuracyQueriesPCA(filename, d):
    data = np.load(filename+"_"+str(d)+'_kmeans_accuracyQueries.npz')
    scores_kmeans, queries_kmeans = data['arr_0'], data['arr_1']
    data = np.load(filename+"_"+str(d)+'_ecc_accuracyQueries.npz')
    queries, scores_ecc = data['arr_2'], data['arr_3']
    queries_ecc_mean = np.nanmean(queries, axis=0)
    scores_ecc_mean = np.nanmean(scores_ecc, axis=0)
    
    avg_queries_kmeans = np.mean(queries_kmeans, axis=0)
    print('#Queries kmeans:', int(np.mean(queries_kmeans, axis=0)))
    print('#Queries ecc:', int(queries_ecc_mean[-1]))
    print('Accuracy kmeans:', np.mean(scores_kmeans, axis=0))
    print('Accuracy ecc:', scores_ecc_mean[-1])
    
    #Plot results
    f = plt.figure(figsize=(7,6))
    plt.hlines(1-np.mean(scores_kmeans, axis=0), avg_queries_kmeans, 
               queries_ecc_mean[-1], linestyle='--', colors='blue', label='SCQ-k-means')
    plt.scatter(avg_queries_kmeans, 1-np.mean(scores_kmeans, axis=0),
                          color='blue', marker='o')
    plt.plot(queries_ecc_mean, 1-scores_ecc_mean, 
             marker='o', color='red', label='RECUR')
# =============================================================================
#     plt.fill_between(queries_ecc_mean, scores_ecc_mean - np.nanstd(scores_ecc, axis = 0)/2, 
#                      scores_ecc_mean + np.nanstd(scores_ecc, axis = 0)/2, 
#                      facecolor='red', alpha=0.5)
# =============================================================================
    
    plt.ylim(0, 1)
    plt.xlim(0, queries_ecc_mean[-1])
    
    plt.xlabel('queries', fontsize=16)
    plt.ylabel('clustering error',  fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, framealpha=1)
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_" + str(d) + "_" + "PCA_errorQueries.eps", bbox_inches='tight')    

def plotAccuracyQueries(filename):
    data = np.load(filename+'8_kmeans_accuracyQueries.npz')
    scores_kmeans, queries_kmeans = data['arr_0'], data['arr_1']
    data = np.load(filename+'8_ecc_accuracyQueries.npz')
    queries, scores_ecc = data['arr_2'], data['arr_3']
    queries_ecc_mean = np.nanmean(queries, axis=0)
    scores_ecc_mean = np.nanmean(scores_ecc, axis=0)
    
    avg_queries_kmeans = np.mean(queries_kmeans, axis=0)
    print('#Queries kmeans:', int(np.mean(queries_kmeans, axis=0)))
    print('#Queries ecc:', int(queries_ecc_mean[-1]))
    print('Accuracy kmeans:', np.mean(scores_kmeans, axis=0))
    print('Accuracy ecc:', scores_ecc_mean[-1])
    
    #Plot results
    f = plt.figure(figsize=(7,6))
    plt.hlines(1-np.mean(scores_kmeans, axis=0), avg_queries_kmeans, 
               queries_ecc_mean[-1], linestyle='--', colors='blue', label='SCQ-k-means')
    plt.scatter(avg_queries_kmeans, 1-np.mean(scores_kmeans, axis=0),
                          color='blue', marker='o')
    plt.plot(queries_ecc_mean, 1-scores_ecc_mean, 
             marker='o', color='red', label='RECUR')
# =============================================================================
#     plt.fill_between(queries_ecc_mean, scores_ecc_mean - np.nanstd(scores_ecc, axis = 0)/2, 
#                      scores_ecc_mean + np.nanstd(scores_ecc, axis = 0)/2, 
#                      facecolor='red', alpha=0.5)
# =============================================================================
    
    plt.ylim(0, 1)
    plt.xlim(0, queries_ecc_mean[-1])
    
    plt.xlabel('queries', fontsize=16)
    plt.ylabel('clustering error',  fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, framealpha=1)
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "8_errorQueries.eps", bbox_inches='tight')


def plotAccuracyBudget(filename):
    data = np.load(filename+'_kmeans_accuracy.npz')
    B, scores_kmeans = data['arr_0'], data['arr_1']
    data = np.load(filename+'_ecc_accuracy.npz')
    B, scores_ecc = data['arr_0'], data['arr_1']
    #Plot results
    f = plt.figure()
    plt.plot(B, 1-np.mean(scores_kmeans, axis = 0), color='blue', label='kmeans')
# =============================================================================
#     plt.fill_between(B, np.mean(scores_kmeans, axis = 0) - np.std(scores_kmeans, axis = 0)/2, 
#                      np.mean(scores_kmeans, axis = 0) + np.std(scores_kmeans, axis = 0)/2, 
#                      facecolor='blue', alpha=0.5)
# =============================================================================
    plt.plot(B, 1-np.mean(scores_ecc, axis = 0), color='red', label='ecc')
# =============================================================================
#     plt.fill_between(B, np.mean(scores_ecc, axis = 0) - np.std(scores_ecc, axis = 0)/2, 
#                      np.mean(scores_ecc, axis = 0) + np.std(scores_ecc, axis = 0)/2, 
#                      facecolor='red', alpha=0.5)
# =============================================================================
    
    plt.ylim(0, 1.1)
    
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_accuracyBudget.jpeg", 
               format="jpeg", bbox_inches='tight')


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
            s=1, 
            marker=',', 
            c=np.array(np.tile(rgb, (X_i.shape[0], 1))), 
            label='cluster ' + str(i+1))
    plt.rcParams.update({'font.size': 15})
    #plt.xlabel(r'$x_1$')
    #plt.ylabel(r'$x_2$')
    plt.grid()
    plt.show()
    f.savefig('Figures/' + filename + "_clustering.png", 
              dpi=600, format="png", bbox_inches='tight')