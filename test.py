import oracle, cleaning
from Dataset import Dataset
import numpy as np, pandas as pd
import qinfer.utils as qut
import scipy.linalg

np.random.seed(0)
n,d,k=100000,2,3
ds=Dataset(n,d,k)
ds.generate()

orcl=oracle.SCQOracle(pd.DataFrame(ds.y_)) # the oracle

X=pd.DataFrame(ds.X_) # our unlabeled dataset
X['y']=np.nan

U=X.copy() # the yet unlabeled elements
gamma=.2

while not U.empty:
    print("%d points left" % U.shape[0])

    # 1. Sample points, check it's not over
#    print("taking sample...")
    s=min(U.shape[0], 10 * k) # sample size
    S=U.sample(s).index # sample
    Sl=np.array([orcl.label(i) for i in S]) # labels of S
    if s==U.shape[0]: # we have labeled all of X
        X.loc[S,'y']=Sl
        break
    X.loc[S,'y']=Sl

    # 2. Take the relative majority cluster C
#    print("finding C...")
    nc=pd.DataFrame({'idx': S, 'lab': Sl}).groupby('lab').count() # number of samples per cluster
    C=nc.sort_values(by='idx',ascending=False).index[0] # id of the largest cluster sample
    SC=S[Sl==C]
    XSC=X.loc[SC].iloc[:,:-1]
#    print("got %d points from cluster %d" % (len(SC), C))

    # 3. Compute the separator, change coordinate system
#    print("computing separator between %d and %d points..." % (len(SC), len(S)-len(SC)))
    M_mve,mu_mve=qut.mvee(XSC.values) # the MVEE
    R_mve=scipy.linalg.sqrtm(M_mve) # rotation/stretch matrix
    Xt=X.copy() # project dataset on the eigenvectors of the MVE
    Xt.iloc[:,:-1]=np.dot(Xt.iloc[:,:-1]-mu_mve,R_mve)
    E=cleaning.Ellipsoid(np.zeros(d),l=1.1*np.ones(d)) # now the separator is the unit ball, with some tolerance
    D=Xt[E.contains(Xt.iloc[:,:-1])] # the subset we're going to clean
    D.loc[SC,'y']=True # in C
    nSC=D.index.intersection(S.difference(SC))
    if not nSC.empty:
        D.loc[nSC,'y']=False # not in C
    print("ellipsoid contains %d points" % D.shape[0])

    # 4. Remove false positives
#    print("removing false positives...")
    cleaner=cleaning.Cleaner(D,E,gamma)
    cleaner.greedyHull()
    cleaner.tessellationClean(orcl)
    print("%d points labeled as %d" %(cleaner.getPositives().shape[0], C))

    # 5. Update labels
    X.loc[cleaner.getPositives().index]=C
    U=X.loc[X['y'].isna()]

print("total queries: %d" % orcl.getCount())
nok=sum(X['y']==ds.y_)
print("accuracy: %d/%d=%2f" % (nok, n, nok/n))
