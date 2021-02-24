#!/usr/bin/env python3
#
# Script to test CubicSpline on a variety of data sets
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

import numpy
import matplotlib.pyplot as plt
from CubicSpline import *

# set numbers of nodes for tests
nvals = [5, 10, 20, 40]

# set function and interval that we will interpolate
def f(x):
    return numpy.array(x+numpy.sin(2*x))
a = 0
b = numpy.pi

# create evaluation points for plots
x = numpy.linspace(a,b,201)

# initialize plots
plt.figure(1)
plt.plot(x, f(x), label="$f(x)$")

# loop over node numbers
for n in nvals:

   print("Testing with n =", n)

   # create the nodes and data
   t = numpy.linspace(a,b,n+1)
   y = f(t)

   # fill p by calling cubic spline routines over the evaluation points
   z = CubicSplineCoeffs(t,y)
   p = CubicSplineEvaluate(t,y,z,x)

   # compute maximum error in interpolation
   err = numpy.linalg.norm(f(x)-p, numpy.inf)

   # add interpolant to plot
   plt.figure(1)
   plt.plot(x, p, label=("$p_{%i}(x)$, error = %.2e" % (n,err)) )

   # add error to plot
   plt.figure(2)
   plt.semilogy(x, numpy.abs(f(x)-p), label=("$|f(x)-p_{%i}(x)|$" % (n)) )


# finalize plots
plt.figure(1)
plt.xlabel("$x$")
plt.ylabel("$f(x)$, $p(x)$")
plt.legend()
plt.title("Natural Cubic Spline Interpolants")

plt.figure(2)
plt.xlabel("$x$")
plt.ylabel("$|f(x) - p(x)|$")
plt.legend()
plt.title("Natural Cubic Spline Error")
plt.show()

# end of script
