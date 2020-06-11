import warnings
import utility
import experiments as exp
import numpy as np

#Suppress Warnings
warnings.filterwarnings("ignore")

rep = 10

experiments = exp.Experiments()

# =============================================================================
# idxs = np.random.choice(n, 1000, False)
# utility.plotClustering(X[idxs, :], k, y[idxs], 'parallel')
# =============================================================================

experiments.expQueriesDimensions(rep = rep)

utility.plotQueriesDimensions('test')