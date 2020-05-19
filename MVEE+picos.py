
# coding: utf-8

# In[210]:

import picos
from picos import Problem, RealVariable, HermitianVariable, SymmetricVariable
import numpy as np


# In[239]:

d = 2
P = Problem()
P.options.solver = "cvxopt"
C = 1*np.array([[-.5,-.5],[0,-.3],[.5,.5]]) # the points in C
NC = 1*np.array([[.7,0]]) # the points not in C
M = SymmetricVariable("M", (2,2)) # the ellipsoid's PSD matrix
P.add_constraint(M>>0)
P.add_list_of_constraints([M|x.reshape(1,d)*x.reshape(d,1)<=1 for x in C]) # include points of C
P.add_list_of_constraints([M|x.reshape(1,d)*x.reshape(d,1)>=1 for x in NC]) # exclude points outside C
P.set_objective("max", picos.DetRootN(M))
sol = P.solve()


# In[241]:

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
S,V=np.linalg.eig(M.value) # eigendecomp
L=1/np.sqrt(S.real) # semiaxes lengths
u,v=V[:,0].real,V[:,1].real # semiaxes basis
print(u,L[0])
print(v,L[1])
fig, ax = plt.subplots()
theta=np.sign(u[1])*(180/np.pi)*arccos(dot(u,np.array([1,0]))/norm(u)) # rotation for E
E=Ellipse(xy=(0,0), width=2*L[0], height=2*L[1], angle=theta, fill=False) # E
ax.add_patch(E)
ax.scatter(C[:,0], C[:,1], c="blue")
ax.scatter(NC[:,0], NC[:,1], c="black")


# In[ ]:



