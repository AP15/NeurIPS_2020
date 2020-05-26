# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:46:29 2020

@author: Marco Bressan
"""

import numpy as np 
import scipy.linalg
import scipy.spatial
from sklearn.datasets import make_moons as moons


def normalize(x, lo: float = 0, hi: float = 1):
    """Normalize an array to a given range.
    Parameters
    ----------
    lo
    hi
    Returns
    -------
    """
    if np.max(x)==np.min(x):
        return lo*x/np.max(x)
    else:
        return lo+(hi-lo)*((x-np.min(x))/(np.max(x)-np.min(x)))


def rsphere(d: int, n: int = 1):
    """Draw vectors from the spherical distribution.
    Parameters
    ----------
    d : int
        dimension
    n : int
        number of samples, default is 1
    Returns
    -------
    """

    P=np.random.normal(size=(n, d))
    return P/np.linalg.norm(P, axis=1).reshape(n, 1)


def rball(d: int, n: int = 1):
    """Draw vectors uniformly from the unit ball.
    Parameters
    ----------
    d : int
        dimension
    n : int
        number of samples, default is 1
    Returns
    -------
    """

    P=np.random.normal(size=(n, d+2), scale=1)
    P=P/np.linalg.norm(P, axis=1).reshape((n, 1))
    return P[:, :d]


def randomPSD(d: int, cn: float = 1.0):
    """Generate a random PSD matrix.
    Parameters
    ----------
    d : int
        Side of the matrix
    cn : float
        Condition number
    Returns
    -------
    X : numpy.ndarray
        A PSD matrix with eigenvectors drawn from the spherical distribution and
        eigenvalues drawn from the uniform distribution. The ratio between the
        max and min eigenvalues is set to the condition number cn.
    """

    S=np.random.uniform(size=d)
    S=normalize(np.random.uniform(size=d),1,max(cn, 1.0)) # the singular values
    Q=scipy.linalg.orth(rsphere(d, d).transpose()).transpose()
    M=np.linalg.multi_dot([Q.transpose(),np.diag(S),Q])
    return scipy.linalg.sqrtm(M).real


def randomCluster(n: int, d: int, r: int = None):
    """Generate a random cluster in R^d with rank r.
    Parameters
    ----------
    n : int
        number of points
    d : int
        dimensionality of ambient space
    r : int
        rank of subspace spanned by the cluster
    Returns
    -------
    X : numpy.ndarray
        An n-by-d array containing the cluster points
    """

    r=d if r is None else r
    P=rball(r, n)
    Pd=np.pad(P, [(0, 0), (0, d-r)], mode='constant')
    Pd/=np.max(np.linalg.norm(Pd, axis=1))
    return Pd


def randomClusterMoons(n: int, d: int, r: int = None):
    """Generate a random cluster in R^d with rank r.
    Parameters
    ----------
    n : int
        number of points
    d : int
        dimensionality of ambient space
    r : int
        rank of subspace spanned by the cluster
    Returns
    -------
    X : numpy.ndarray
        An n-by-d array containing the cluster points
    """

    r=d if r is None else r
    P, y = moons(n_samples=n, noise=0.00)
    P = P[y==0]
    Pd=np.pad(P, [(0, 0), (0, d-r)], mode='constant')
    Pd/=np.max(np.linalg.norm(Pd, axis=1))
    return Pd


def toLatent(X, W, c):
    """Map to the latent space.
    Parameters
    ----------
    X: numpy.ndarray
        An n-by-d array, each row is a point in R^d
    W: numpy.ndarray
        A d-by-d matrix specifying the transformation
    c: numpy.ndarray
        A d-size array giving the center of the transformation
    Returns
    -------
    The inner product of W and (X-c). In other words, set the origin in c and then apply W.
    Note that toLatent(toVisible(X,W,c),W,C) gives X.
    """
    return np.dot(X-c, W) if X.shape[0] else np.zeros((0, X.shape[1]))


def toVisible(X, W, c):
    """Reverse of toLatent(X, W, C).
    Parameters
    ----------
    X: numpy.ndarray
        An n-by-d array, each row is a point in R^d
    W: numpy.ndarray
        A d-by-d matrix specifying the transformation
    c: numpy.ndarray
        A d-size array giving the center of the transformation
    Returns
    -------
    The inner product of W^(-1) and X, plus c. In other words, apply the inverse of W and shift by c.
    Note that toVisible(toLatent(X,W,c),W,C) gives X.
    """
    return np.dot(X, np.linalg.inv(W))+c if X.shape[0] else np.zeros((0, X.shape[1]))

def clusterMargins(X, y, Ws, cs):
    """Compute the (latent) margins of the clusters.
    Parameters
    ----------
    X
    y
    Ws
    cs
    Returns
    -------
    For each cluster, the ratio between the closest external point and the cluster's radius.
    """
    g=np.zeros(len(Ws))
    d=X.shape[1]
    for i in range(len(Ws)):
        Xt=toLatent(X, Ws[i], cs[i])
        c=toLatent(cs[i], Ws[i], cs[i])
        C, nC=Xt[y==i], Xt[y!=i]
        radius=np.max(scipy.spatial.distance_matrix(c.reshape(1, d), C))
        dist=np.min(scipy.spatial.distance_matrix(c.reshape(1, d), nC))
        g[i]=dist/radius
    return g

def randomDataset(n: int, k: int, d: int, gamma: float = 0.5, r: int = None, cn: float = None,
                  tightMargin: bool = False):
    """Generate a random dataset controlling the clusters' rank, margin, and stretch.
    Parameters
    ----------
    n : int
        Number of points
    k : int
        Number of clusters
    d : int
        Dimension of ambient space
    gamma : float
        Margin
    r : int
        Rank of clusters
    cn : float
        Condition number of the PSD matrices of clusters. A higher value means higher stretch.
    tightMargin : bool
        Whether to tighten the margin around gamma (default is False)
    Returns
    -------
    X : numpy.ndarray
        an n-by-d array of the input points
    y : numpy.ndarray
        a length-h array containing the cluster labels
    Ws : list
        list of PSD matrices, one per cluster
    cs : list
        list of centers, one per cluster
    """

    if r is None:
        r=d
    X=np.zeros((0, d))
    y=np.zeros(0)
    Ws=[randomPSD(d) if cn is None else randomPSD(d, cn) for i in range(k)]  # random PSD matrices
    cs=rball(d, k)  # random centers
    for i in range(k):
        nk=int(n/k)#(n-X.shape[0])//(k-i)  # cluster size
        y=np.r_[y, i*np.ones(nk)]
        # move to the latent metric of C
        csLat=toLatent(cs, Ws[i], cs[i])
        XLat=toLatent(X, Ws[i], cs[i])  # center and transform
        # distance to other points, including centers of future clusters
        D=np.linalg.norm(np.r_[XLat, csLat[i+1:]], axis=1)
        gap=np.min(D)  # distance of closest point
        # generate cluster
        P=randomCluster(nk, d, r)
        #P*=2*gap/np.max(np.linalg.norm(P, axis=1))  # rescale a bit
        P*=gap/10/d/np.max(np.linalg.norm(P, axis=1)) # rescale a bit
        # convert cluster to visible space and add it
        X=np.r_[X, toVisible(P, Ws[i], cs[i])]  # if X.shape[0]>0 else P
    y=y.astype(int)

    # Adjust margins if too small
    m=clusterMargins(X, y, Ws, cs)

    while min(m)<np.sqrt(1+gamma):
        i=np.argmin(m)
        X[y==i]=toVisible(toLatent(X[y==i], Ws[i], cs[i])*(0.99*m[i])/np.sqrt(1+gamma), Ws[i], cs[i])
        m=clusterMargins(X, y, Ws, cs)

    if tightMargin:
        while max(m)>1.01*np.sqrt(1+gamma):
            i=np.argmax(m)
            X[y==i]=toVisible(toLatent(X[y==i], Ws[i], cs[i])*m[i]/np.sqrt(1+gamma), Ws[i], cs[i])
            m=clusterMargins(X, y, Ws, cs)

    while min(m)<np.sqrt(1+gamma):
        i=np.argmin(m)
        X[y==i]=toVisible(toLatent(X[y==i], Ws[i], cs[i])*(0.99*m[i])/np.sqrt(1+gamma), Ws[i], cs[i])
        m=clusterMargins(X, y, Ws, cs)

    return X, y.astype(int), Ws, cs


def randomDatasetMoons(n: int, k: int, d: int, gamma: float = 0.5, r: int = None, cn: float = None,
                  tightMargin: bool = False):
    """Generate a random dataset controlling the clusters' rank, margin, and stretch.
    Parameters
    ----------
    n : int
        Number of points
    k : int
        Number of clusters
    d : int
        Dimension of ambient space
    gamma : float
        Margin
    r : int
        Rank of clusters
    cn : float
        Condition number of the PSD matrices of clusters. A higher value means higher stretch.
    tightMargin : bool
        Whether to tighten the margin around gamma (default is False)
    Returns
    -------
    X : numpy.ndarray
        an n-by-d array of the input points
    y : numpy.ndarray
        a length-h array containing the cluster labels
    Ws : list
        list of PSD matrices, one per cluster
    cs : list
        list of centers, one per cluster
    """

    if r is None:
        r=d
    X=np.zeros((0, d))
    y=np.zeros(0)
    Ws=[randomPSD(d) if cn is None else randomPSD(d, cn) for i in range(k)]  # random PSD matrices
    cs=rball(d, k)  # random centers
    for i in range(k):
        nk=int(n/k)#(n-X.shape[0])//(k-i)  # cluster size
        y=np.r_[y, i*np.ones(nk)]
        # move to the latent metric of C
        csLat=toLatent(cs, Ws[i], cs[i])
        XLat=toLatent(X, Ws[i], cs[i])  # center and transform
        # distance to other points, including centers of future clusters
        D=np.linalg.norm(np.r_[XLat, csLat[i+1:]], axis=1)
        gap=np.min(D)  # distance of closest point
        # generate cluster
        P=randomClusterMoons(2*nk, d, r)
        #P*=2*gap/np.max(np.linalg.norm(P, axis=1))  # rescale a bit
        P*=gap/10/d/np.max(np.linalg.norm(P, axis=1)) # rescale a bit
        # convert cluster to visible space and add it
        X=np.r_[X, toVisible(P, Ws[i], cs[i])]  # if X.shape[0]>0 else P
    y=y.astype(int)

    # Adjust margins if too small
    m=clusterMargins(X, y, Ws, cs)

    while min(m)<np.sqrt(1+gamma):
        i=np.argmin(m)
        X[y==i]=toVisible(toLatent(X[y==i], Ws[i], cs[i])*(0.99*m[i])/np.sqrt(1+gamma), Ws[i], cs[i])
        m=clusterMargins(X, y, Ws, cs)

    if tightMargin:
        while max(m)>1.01*np.sqrt(1+gamma):
            i=np.argmax(m)
            X[y==i]=toVisible(toLatent(X[y==i], Ws[i], cs[i])*m[i]/np.sqrt(1+gamma), Ws[i], cs[i])
            m=clusterMargins(X, y, Ws, cs)

    while min(m)<np.sqrt(1+gamma):
        i=np.argmin(m)
        X[y==i]=toVisible(toLatent(X[y==i], Ws[i], cs[i])*(0.99*m[i])/np.sqrt(1+gamma), Ws[i], cs[i])
        m=clusterMargins(X, y, Ws, cs)

    return X, y.astype(int), Ws, cs


if __name__ == '__main__':
    # Sample usage
    import matplotlib.pyplot as plt
    print("generating clusters...")
    X,y,Ws,cs=randomDataset(n=1000, k=5, d=2, r=2, gamma=.5, cn=10, tightMargin=True)
    fig=plt.figure(figsize=(5, 5))
    fig.add_subplot(111).scatter(X[:, 0], X[:, 1], c=y, s=1)
    if False: # plot each cluster in its latent space
        for i in np.unique(y):
            Xt=toLatent(X, Ws[i], cs[i])
            fig=plt.figure()
            ax=fig.add_subplot(111)
            C=Xt[y==i]
            ax.scatter(Xt[:, 0], Xt[:, 1], c=y, s=3)
            ax.set_xlim(4*C[:, 0].min(), 4*C[:, 0].max())
            ax.set_ylim(4*C[:, 1].min(), 4*C[:, 1].max())
    plt.show()