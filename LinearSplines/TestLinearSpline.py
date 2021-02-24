#!/usr/bin/env python3
#
# Script to test LinearSpline on a variety of data sets
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from LinearSpline import LinearSpline

# set numbers of nodes for tests
nvals = [5, 10, 20, 40]

# set function and interval that we will interpolate
def f(x):
    return numpy.array(x+numpy.sin(2*x))
a = 0
b = 4

# create evaluation points for plots
x = numpy.linspace(a,b,201)

# initialize plot
plt.figure(1)
plt.plot(x, f(x), label="$f(x)$")

# loop over node numbers
for n in nvals:

   print("Testing with n =", n)

   # create the nodes and data
   t = numpy.linspace(a,b,n+1)
   y = f(t)

   # fill p by calling LinearSpline over the evaluation points
   p = LinearSpline(x,t,y)

   # compute maximum error in interpolation
   err = numpy.linalg.norm(f(x)-p, numpy.inf)

   # add interpolant to plot
   plt.figure(1)
   plt.plot(x, p, label=("$p_{%i}(x)$, error = %.2e" % (n,err)) )

# finalize plot
plt.figure(1)
plt.xlabel("$x$")
plt.ylabel("$f(x)$, $p(x)$")
plt.legend()
plt.title("Linear Spline Interpolant Demo")
plt.show()

# end of script
