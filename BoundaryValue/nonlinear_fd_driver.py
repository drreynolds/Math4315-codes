#!/usr/bin/env python3
#
# Demo to show second-order collocation method for the nonlinear BVP
#     u'' = 3(u')^2/u, -1<x<2
#     u(-1) = 1,  u(2) = 1/2
# using various uniform mesh sizes.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from differentiation_matrix import *
from newton import *

# set numbers of intervals for tests
nvals = [10, 20, 40, 80, 160, 320]

# setup problem, analytical solution, etc
a = -1.0
b = 2.0
alpha = 1.0
beta = 0.5
def utrue(x):
    return (1.0/numpy.sqrt(x+2.0))

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

    # set nonlinear root-finding function and Jacobian
    def phi(u):
        return (3.0*((Dx@u)**2)/u)
    def Jphi(u):
        return (6.0*numpy.diag((Dx@u)/u)@Dx - 3.0*numpy.diag(((Dx@u)/u)**2))
    E = numpy.zeros((n-1,n+1),dtype=float)
    E[:,1:-1] = numpy.eye(n-1)
    def f(u):
        res = numpy.zeros(n+1)
        res[0] = u[0]-alpha
        res[-1] = u[-1]-beta
        res[1:-1] = E@(Dxx@u-phi(u))
        return res
    def J(u):
        Jac = numpy.zeros((n+1,n+1),dtype=float)
        Jac[0,0] = 1.0
        Jac[-1,-1] = 1.0;
        Jac[1:-1,:] = E@(Dxx - Jphi(u))
        return Jac

    # set initial condition to satisfy boundary conditions
    u = 1.0 - 1.0/6.0*(numpy.linspace(-1,2,n+1)+1.0)

    # call Newton method to solve root-finding problem
    print('Calling Newton method to solve for mesh size n = %i:' % (n))
    u, its = newton(f, J, u, 20, 1e-9, 1e-11, 1)

    # compute error and h
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
plt.title('Second-order Nonlinear FD Approximations')

plt.figure(2)
plt.xlabel('$x$')
plt.ylabel('$|\hat{u}(x) - u(x)|$')
plt.legend()
plt.title('Second-order Nonlinear FD Error')

plt.show()

# end of script
