# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:39:50 2020

@author: apaudice
"""

# Convex Optimization, S. Boyd, L. Vandeberghe 
# Figures 8.3 and 8.4, pages 412 and 416.
# Ellipsoidal approximations.

from math import log, pi
import numpy as np
from cvxopt import blas, lapack, solvers, matrix, mul, cos, sin
solvers.options['show_progress'] = False
try: import pylab
except ImportError: pylab_installed = False
else: pylab_installed = True

# Extreme points (with first one appended at the end)
#X = matrix([ 0.55,  0.25, -0.20, -0.25,  0.00,  0.40,  0.55,
#             0.00,  0.35,  0.20, -0.10, -0.30, -0.20,  0.00 ], (7,2))

n = 100

Y = np.random.rand(n, 2)
Z = np.asmatrix(Y[0, :])
Y = np.concatenate((Y, Z))
X = matrix(Y)

m = X.size[0] - 1

# Inequality description G*x <= h with h = 1
G, h = matrix(0.0, (m,2)), matrix(0.0, (m,1))
G = (X[:m,:] - X[1:,:]) * matrix([0., -1., 1., 0.], (2,2))
h = (G * X.T)[::m+1]
G = mul(h[:,[0,0]]**-1, G)
h = matrix(1.0, (m,1))

# Loewner-John ellipsoid
#
# minimize     log det A^-1
# subject to   xk'*A*xk - 2*xk'*b + b'*A^1*b <= 1,  k=1,...,m
#
# 5 variables x = (A[0,0], A[1,0], A[1,1], b[0], b[1])

def F(x=None, z=None):
    if x is None:
        return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])

    # Factor A as A = L*L'.  Compute inverse B = A^-1.
    A = matrix( [x[0], x[1], x[1], x[2]], (2,2))
    L = +A
    try: lapack.potrf(L)
    except: return None
    B = +L
    lapack.potri(B)
    B[0,1] = B[1,0]

    # f0 = -log det A
    f = matrix(0.0, (m+1,1))
    f[0] = -2.0 * (log(L[0,0]) + log(L[1,1]))

    # fk = xk'*A*xk - 2*xk'*b + b*A^-1*b - 1
    #    = (xk - c)' * A * (xk - c) - 1  where c = A^-1*b
    c = x[3:]
    lapack.potrs(L, c)
    for k in range(m):
        f[k+1] = (X[k,:].T - c).T * A * (X[k,:].T - c) - 1.0

    # gradf0 = (-A^-1, 0) = (-B, 0)
    Df = matrix(0.0, (m+1,5))
    Df[0,0], Df[0,1], Df[0,2] = -B[0,0], -2.0*B[1,0], -B[1,1]

    # gradfk = (xk*xk' - A^-1*b*b'*A^-1,  2*(-xk + A^-1*b))
    #        = (xk*xk' - c*c', 2*(-xk+c))
    Df[1:,0] = X[:m,0]**2 - c[0]**2
    Df[1:,1] = 2.0 * (mul(X[:m,0], X[:m,1]) - c[0]*c[1])
    Df[1:,2] = X[:m,1]**2 - c[1]**2
    Df[1:,3] = 2.0 * (-X[:m,0] + c[0])
    Df[1:,4] = 2.0 * (-X[:m,1] + c[1])

    if z is None: return f, Df

    # hessf0(Y, y) = (A^-1*Y*A^-1, 0) = (B*YB, 0)
    H0 = matrix(0.0, (5,5))
    H0[0,0] = B[0,0]**2
    H0[1,0] = 2.0 * B[0,0] * B[1,0]
    H0[2,0] = B[1,0]**2
    H0[1,1] = 2.0 * ( B[0,0] * B[1,1] + B[1,0]**2 )
    H0[2,1] = 2.0 * B[1,0] * B[1,1]
    H0[2,2] = B[1,1]**2

    # hessfi(Y, y)
    #     = ( A^-1*Y*A^-1*b*b'*A^-1 + A^-1*b*b'*A^-1*Y*A^-1
    #             - A^-1*y*b'*A^-1 - A^-1*b*y'*A^-1,
    #         -2*A^-1*Y*A^-1*b + 2*A^-1*y )
    #     = ( B*Y*c*c' + c*c'*Y*B - B*y*c' - c*y'*B,  -2*B*Y*c + 2*B*y )
    #     = ( B*(Y*c-y)*c' + c*(Y*c-y)'*B, -2*B*(Y*c - y) )
    H1 = matrix(0.0, (5,5))
    H1[0,0] = 2.0 * c[0]**2 * B[0,0]
    H1[1,0] = 2.0 * ( c[0] * c[1] * B[0,0] + c[0]**2 * B[1,0] )
    H1[2,0] = 2.0 * c[0] * c[1] * B[1,0]
    H1[3:,0] = -2.0 * c[0] * B[:,0]
    H1[1,1] = 2.0 * c[0]**2 * B[1,1] + 4.0 * c[0]*c[1]*B[1,0]  + \
              2.0 * c[1]**2 + B[0,0]
    H1[2,1] = 2.0 * (c[1]**2 * B[1,0] + c[0]*c[1]*B[1,1])
    H1[3:,1] = -2.0 * B * c[[1,0]]
    H1[2,2] = 2.0 * c[1]**2 * B[1,1]
    H1[3:,2] = -2.0 * c[1] * B[:,1]
    H1[3:,3:] = 2*B

    return f, Df, z[0]*H0 + sum(z[1:])*H1

sol = solvers.cp(F)
A = matrix( sol['x'][[0, 1, 1, 2]], (2,2))
b = sol['x'][3:]

if pylab_installed:
    pylab.figure(1, facecolor='w')
    pylab.plot(X[:,0], X[:,1], 'ko')

    # Ellipsoid in the form { x | || L' * (x-c) ||_2 <= 1 }
    L = +A
    lapack.potrf(L)
    c = +b
    lapack.potrs(L, c)

    # 1000 points on the unit circle
    nopts = 1000
    angles = matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
    circle = matrix(0.0, (2,nopts))
    circle[0,:], circle[1,:] = cos(angles), sin(angles)

    # ellipse = L^-T * circle + c
    blas.trsm(L, circle, transA='T')
    ellipse = circle + c[:, nopts*[0]]
    ellipse2 = 0.5 * circle + c[:, nopts*[0]]

    pylab.plot(ellipse[0,:].T, ellipse[1,:].T, 'k-')
    pylab.fill(ellipse2[0,:].T, ellipse2[1,:].T, facecolor = '#F0F0F0')
    pylab.title('Loewner-John ellipsoid')
    pylab.axis('equal')
    pylab.axis('off')