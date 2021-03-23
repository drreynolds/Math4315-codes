#!/usr/bin/env python3
#
# Script to plot cardinal functions for polynomial interpolants using various choices of node types.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from Lagrange import Lagrange

# set numbers of nodes for tests
nvals = [5, 10, 15, 20]

# set interval that we will interpolate over
a = -1
b = 1

# set evaluation points to compare interpolants
x = numpy.linspace(a, b, 2001)

# loop over node numbers
for n in nvals:

    print("Plotting cardinal functions for n =", n)

    # set (n+1)x(n+1) identity matrix
    I = numpy.eye(n+1)

    # create figure window for this n
    plt.figure()

    # uniformly-spaced nodes
    plt.subplot(1,3,1)
    t = numpy.linspace(a,b,n+1)
    for i in range(n+1):
        p = Lagrange(t, I[i,:], x)
        plt.plot(x, p)
    plt.xlabel('$x$')
    plt.ylabel('$I(e_{k})$')
    plt.title('Uniformly spaced nodes, n = %i' % (n))

    # random nodes
    plt.subplot(1,3,2)
    t = a + (b-a)*numpy.random.rand(n+1)
    for i in range(n+1):
        p = Lagrange(t, I[i,:], x)
        plt.plot(x, p)
    plt.xlabel('$x$')
    plt.ylabel('$I(e_{k})$')
    plt.title('Random nodes, n = %i' % (n))

    # Chebyshev nodes
    plt.subplot(1,3,3)
    t = (a+b)/2 - (b-a)/2*numpy.cos(numpy.linspace(0,n,n+1)*numpy.pi/n)
    for i in range(n+1):
        p = Lagrange(t, I[i,:], x)
        plt.plot(x, p)
    plt.xlabel('$x$')
    plt.ylabel('$I(e_{k})$')
    plt.title('Chebyshev nodes, n = %i' % (n))

plt.show()

# end of script
