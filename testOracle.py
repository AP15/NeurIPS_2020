# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:16:08 2020

@author: apaudice
"""

import oracle
import pandas as pd
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

batgirl = oracle.SCQOracle(df)

#Test scq
print(df[4][0], df[4][30])
print(batgirl.scq(1, 30))
print(batgirl.scq(5, 70))
print(batgirl.scq(15, 90))
print(batgirl.scq(0, 1))

#Test getCount
print('Count: ', batgirl.getCount())

#Test label
print(batgirl.label(2))
print('Count: ', batgirl.getCount())

print(batgirl.label(80))
print('Count: ', batgirl.getCount())

print(batgirl.label(120))
print('Count: ', batgirl.getCount())