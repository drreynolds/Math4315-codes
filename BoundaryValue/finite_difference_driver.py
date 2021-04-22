#!/usr/bin/env python3
#
# Demo to show second-order collocation method for BVP
#     u'' + (1/(2+x)) u' + (11x/(2+x)) u = (-e^x (12x^3 + 7x^2 + 1))/(2+x), -1<x<1
#     u(-1) = u(1) = 0
# using various uniform mesh sizes.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from differentiation_matrix import *

### utility functions ###

def enforce_boundary(A,rhs):
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
    rhs[0] = 0
    rhs[-1] = 0
    return [A, rhs]


### main script ###

# set numbers of intervals for tests
nvals = [10, 20, 40, 80, 160, 320]

# setup problem, analytical solution, etc
a = -1.0
b = 1.0
def p(x):
    return (1.0/(2+x))
def q(x):
    return (11.0*x/(2+x))
def r(x):
    return (-numpy.exp(x)*(12*x**3 + 7*x**2 + 1)/(2+x))
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

    # get partition and differentiation matrices
    t, Dx  = differentiation_matrix(n, a, b, 2, 1)
    t, Dxx = differentiation_matrix(n, a, b, 2, 2)

    # create additional diagonal matrices
    P = numpy.diag(p(t))
    Q = numpy.diag(q(t))
    rhs = r(t)
    A, rhs = enforce_boundary(Dxx + P@Dx + Q, rhs)

    # solve linear system, compute error and h
    u = numpy.linalg.solve(A, rhs)
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
plt.title('Second-order FD Approximations')

plt.figure(2)
plt.xlabel('$x$')
plt.ylabel('$|\hat{u}(x) - u(x)|$')
plt.legend()
plt.title('Second-order FD Approximation Error')

plt.show()

# end of script


# end of script
