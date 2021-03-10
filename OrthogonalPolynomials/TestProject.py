#!/usr/bin/env python3
#
# Script to compare orthogonal polynomial projections.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from Lagrange import Lagrange
from Legendre import Legendre
from Chebyshev import Chebyshev
from L2InnerProduct import L2InnerProduct

# set interval and weight functions
a = -1.0
b = 1.0
def wL(x):
    if numpy.isscalar(x):
        return 1.0
    else:
        return numpy.ones(x.shape)
def wC(x):
    return 1.0/numpy.sqrt(1.0-x**2)

# set numbers of nodes for tests
nvals = [5, 10, 15, 20]

# set function that we will interpolate
def f(x):
    return 1/(1+x**2)

# set evaluation points for plots
x = numpy.linspace(a, b, 2001)

# initialize plots
plt.figure(1)    # Chebyshev interpolant
plt.plot(x, f(x), label='$f(x)$')
plt.figure(3)    # Legendre projection
plt.plot(x, f(x), label='$f(x)$')
plt.figure(5)    # Chebyshev projection
plt.plot(x, f(x), label='$f(x)$')

# loop over node numbers
for n in nvals:

   print("Testing with n =", n)

   # Chebyshev interpolant
   plt.figure(1)
   t = numpy.cos((2*numpy.linspace(0,n,n+1)+1)/(2*n+2)*numpy.pi)
   p = Lagrange(t, f(t), x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(2)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))

   # Legendre projection
   plt.figure(3)
   p = numpy.zeros(x.shape)
   for k in range(n+1):
       def pk(x):
           return Legendre(x,k);
       c = L2InnerProduct(pk,f,wL,a,b) * (k+0.5)
       p += c*pk(x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(4)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))

   # Chebyshev projection
   plt.figure(5)
   p = numpy.zeros(x.shape)
   for k in range(n+1):
       def pk(x):
           return Chebyshev(x,k);
       if (k==0):
           c = L2InnerProduct(pk,f,wC,a,b) / numpy.pi
       else:
           c = L2InnerProduct(pk,f,wC,a,b) * 2.0 / numpy.pi
       p += c*pk(x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(6)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))


# finalize plots
plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Chebyshev interpolant')

plt.figure(2)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Chebyshev interpolant error')

plt.figure(3)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Legendre projection')

plt.figure(4)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Legendre projection error')

plt.figure(5)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Chebyshev projection')

plt.figure(6)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Chebyshev projection error')

plt.show()

# end of script
