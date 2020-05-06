from Dataset import Dataset
import numpy as np, pandas as pd
import scipy.linalg
from scipy.stats import special_ortho_group

def normalize(x, lo:float=0, hi:float=1):
    """Normalize an array.

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

def randomSpherical(d:int, n:int=1):
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

    P=np.random.normal(size=(n,d))
    return P/np.linalg.norm(P,axis=1).reshape(n,1)

def randomBall(d:int, n:int=1):
    """Draw vectors from the uniform ball distribution.

    Parameters
    ----------
    d : int
        dimension
    n : int
        number of samples, default is 1

    Returns
    -------

    """

    P=np.random.normal(size=(n,d+2), scale=1)
    P=P/np.linalg.norm(P,axis=1).reshape((n,1))
    return P[:,:d]

def randomPSD(d:int, cn:float=100.0):
    """Generate a random PSD matrix.

    Parameters
    ----------
    d : int
        side of the matrix
    cn : float
        condition number

    Returns
    -------
    X : numpy.ndarray
        the matrix
    """

    U=randomSpherical(d,d)
    cn=max(cn,1.0)
    S=np.random.uniform(size=d) # the singular values
    S=normalize(S,1,np.sqrt(cn))
    M=np.linalg.multi_dot([U.transpose(),np.diag(S),U])
    return scipy.linalg.sqrtm(M)

def randomCluster(n:int, d:int, r:int=None):
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
        an n-by-d array containing the cluster points
    """

    if r is None:
        r=d
    P=np.random.normal(size=(n,r+2), scale=1)
    P=P/np.linalg.norm(P,axis=1).reshape((n,1))
    P=P[:,:r]
#    P=np.random.multivariate_normal(np.zeros(r), np.ones(r), n)
    Pd=np.pad(P, [(0, 0), (0, d-r)], mode='constant')
    return Pd

def randomDataset(n:int, k:int, d:int, gamma:float=0.5, r:int=None):
    """Generate a random dataset.

    Parameters
    ----------
    n : int
        number of points
    k : int
        number of clusters
    d : int
        dimension of ambient space
    gamma : float
        margin
    r : int
        rank of clusters

    Returns
    -------
    X : numpy.ndarray
        an n-by-d array of the input points
    y : numpy.ndarray
        a length-h array containing the cluster labels
    Ws : list
        list of PSD patrices, one per cluster
    cs : list
        list of centers, one per cluster
    """

    X=np.zeros((0,d))
    y=np.zeros(0)
    Ws=[] # matrices
    cs=np.zeros((0,d)) # centers
    for i in range(k):
        # generate cluster C, centered at 0 with radius ≤ 1
        nk=n//k
        P=randomCluster(nk,d,r)
#        print(P)
        print(max(np.linalg.norm(P,axis=1)))
        P=P-np.mean(P,axis=0)
        P=P/max(np.linalg.norm(P,axis=1))
#        print(np.where(P[:,0]>1))
        # generate PSD matrix of C
        W=randomPSD(d)
        Ws+=[W]
        # transform space with the latent metric of C
        if X.shape[0]:
            X,cs=np.dot(X,W),np.dot(cs,W)
            x=X[np.argmin(X,axis=0)[0]]
            # shift so that the point closest to the origin is at sqrt(1+gamma)*e1
            X=X-x+np.sqrt(1+gamma)*np.eye(1,d,0)
            cs=cs-x+np.sqrt(1+gamma)*np.eye(1,d,0)
        # insert cluster
        X=np.r_[X,P] if X.shape[0] else P
        cs=np.r_[cs,np.zeros((1,d))] if cs.shape[0] else np.zeros((1,d))
        # transform dataset back to original coordinates
        X=np.dot(X,np.linalg.inv(W))
        cs=np.dot(cs,np.linalg.inv(W))
        y=np.r_[y,i*np.ones(nk)]
    return X, y.astype(int), Ws, cs
