#!/usr/bin/env python3
#
# Demo to show piecewise linear finite element solver for BVP
#     [(2+x) u']' + 11xu = -e^x (12x^3 + 7x^2 + 1), -1<x<1
#     u(-1) = u(1) = 0
# using various non-uniform meshes.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt

### utility functions ###

def partition(a, b, n):
    """
    Utility routine to construct the non-uniform partition, t, for the
    interval [a,b] using (n+1) nodes.  For convenience I just use the
    Chebyshev nodes of the second kind.  Returns t as a column vector.
    """
    return ((a+b)/2 - (b-a)/2*numpy.cos(numpy.linspace(0,n,n+1)*numpy.pi/n))

def stiffness_matrix(c, t):
    """
    Utility routine to construct the stiffness matrix, K, based on the
    coefficient function c(x) and partition t.  This matrix includes rows
    for the boundary nodes.
    """
    Ke = numpy.array([[1, -1], [-1, 1]], dtype=float)
    n = t.size-1
    h = t[1:]-t[:-1]
    cvals = c(t)
    cavg = 0.5*(cvals[1:]+cvals[:-1])
    K = numpy.zeros((n+1,n+1), dtype=float)
    K[1,1] = cavg[0]/h[0]
    K[n-1,n-1] = cavg[n-1]/h[n-1]
    for i in range(2,n):
        K[i-1:i+1,i-1:i+1] += (cavg[i-1]/h[i-1])*Ke
    return K

def mass_matrix(s, t):
    """
    Utility routine to construct the mass matrix, M, based on the
    coefficient function s(x) and partition t.  This matrix includes rows
    for the boundary nodes.
    """
    Me = (1.0/6.0)*numpy.array([[2, 1], [1, 2]], dtype=float)
    n = t.size-1
    h = t[1:]-t[:-1]
    svals = s(t)
    savg = 0.5*(svals[1:]+svals[:-1])
    M = numpy.zeros((n+1,n+1), dtype=float)
    M[1,1] = savg[0]*h[0]/3
    M[n-1,n-1] = savg[n-1]*h[n-1]/3
    for i in range(2,n):
        M[i-1:i+1,i-1:i+1] += (savg[i-1]*h[i-1])*Me
    return M

def rhs_vector(f, t):
    """
    Utility routine to construct the right-hand side vector, b, based on the
    forcing function f(x) and partition t.  This vector includes entries
    for the boundary nodes.
    """
    fe = 0.5*numpy.array([1, 1], dtype=float)
    n = t.size-1
    h = t[1:]-t[:-1]
    fvals = f(t)
    favg = 0.5*(fvals[1:]+fvals[:-1])
    r = numpy.zeros(n+1)
    r[1] = favg[0]*h[0]/2
    r[n-1] = favg[n-1]*h[n-1]/2
    for i in range(2,n):
        r[i-1:i+1] += (favg[i-1]*h[i-1])*fe
    return r

def enforce_boundary(A,r):
    """
    Utility routine to enforce the homogeneous Dirichlet boundary conditions on
    the linear system encoded in the matrix A and right-hand side vector r.
    This routine assumes that the inputs include placeholder rows for these
    conditions.
    """
    A[0,:] = 0*A[0,:]
    A[-1,:] = 0*A[-1,:]
    A[0,0] = numpy.max(numpy.abs(numpy.diag(A)))
    A[-1,-1] = numpy.max(numpy.abs(numpy.diag(A)))
    r[0] = 0
    r[-1] = 0
    return [A, r]


### main script ###

# set numbers of intervals for tests
nvals = [10, 20, 40, 80, 160, 320]

# setup problem, analytical solution, etc
a = -1.0
b = 1.0
def c(x):
    return (-(2.0+x))
def s(x):
    return (11.0*x)
def f(x):
    return (-numpy.exp(x)*(12*x**3 + 7*x**2 + 1))
def utrue(x):
    return (numpy.exp(x)*(1-x**2))

# initialize plots
plt.figure(1)
x = numpy.linspace(a,b,1000)
plt.plot(x, utrue(x), label='$\hat{u}(x)$')

# initialize 'current' mesh size and error norm
h_cur = 1.0
e_cur = 1.0

# loop over partition sizes
for n in nvals:

    # get partition
    t  = partition(a, b, n)

    # construct linear system
    K = stiffness_matrix(c, t)
    M = mass_matrix(s, t)
    r = rhs_vector(f, t)
    A, r = enforce_boundary(K+M, r)

    # solve linear system, compute error and h
    u = numpy.linalg.solve(A, r)
    e = numpy.abs(u-utrue(t))
    e_old = e_cur
    e_cur = numpy.linalg.norm(e,numpy.Inf)
    h_old = h_cur
    h_cur = numpy.max(t[1:]-t[:-1])

    # update plots
    plt.figure(1)
    plt.plot(t, u, label=('$u_{%i}(x)$' % (n)))
    plt.figure(2)
    plt.semilogy(t, e, label=('$|\hat{u}-u_{%i}|$' % (n)))

    # output current error norm and estimated convergence rate
    if (n > nvals[0]):
        print('n = %3i,  h = %.2e, ||error|| = %.2e,  conv. rate = %.2f' %
              (n, h_cur, e_cur, numpy.log(e_cur/e_old)/numpy.log(h_cur/h_old)))
    else:
        print('n = %3i,  h = %.2e, ||error|| = %.2e' % (n, h_cur, e_cur))


# finalize plots
plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.legend()
plt.title('P1 FEM Approximations')

plt.figure(2)
plt.xlabel('$x$')
plt.ylabel('$|\hat{u}(x) - u(x)|$')
plt.legend()
plt.title('P1 FEM Approximation Error')

plt.show()

# end of script
