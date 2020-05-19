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

utility.experiment1('general', 100000, 2, 7, 10)
utility.plotAccuracyBudget('general')