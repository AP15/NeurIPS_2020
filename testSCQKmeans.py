# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:51:27 2020

@author: apaudice
"""

import utility

#utility.experiment1('aggregation', rep = 1)
#utility.plotAccuracyBudget('aggregation')

#utility.experiment1('parallel', 100000, 2, 10, 1)
#utility.plotAccuracyBudget('parallel')

#utility.experiment1kmeans('general', 100000, 2, 10, 10)
#utility.plotAccuracyBudget('general')

utility.experiment2kmeans('general', 1000, 2, 3, 3)
utility.scatterAccuracyGamma('general')
