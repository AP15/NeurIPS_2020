import oracle
import numpy as np
import pandas as pd
import datagen
import LabelSampler as ls

n, d, k, g = 500, 1, 3, 1
X_, y_, Ws, cs = datagen.randomDataset(n, k, d, g)
X = pd.DataFrame(X_) 
X['y'] = np.nan
oc = oracle.SCQOracle(pd.DataFrame(y_))

for t in [1,2,10,100]: # thresholds
    sampler = ls.LabelSampler(oc, t)
    while X.iloc[:,-1].isna().any():
        U = X[X.iloc[:,-1].isna()]
        l,s = sampler.sample(U)
#        print(l,s)
        assert (y_[s] == l).all(), "Sampled labels do not match ground truth!"
        assert (np.array([idx in U.index for idx in s])).all(), "Some samples are outside the specified dataset!"
        X.iloc[s,-1] = l
