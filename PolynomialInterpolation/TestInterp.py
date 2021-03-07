#!/usr/bin/env python3
#
# Script to compare polynomial interpolants using various choices of node types.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from Lagrange import Lagrange

# set numbers of nodes for tests
nvals = [5, 10, 20, 40]

# set function and interval that we will interpolate
def f(x):
    return 1/(1+x**2)
a = -5
b = 5

# set evaluation points to compare interpolants
x = numpy.linspace(a, b, 2001)

# initialize plots
plt.figure(1)    # uniform nodes
plt.plot(x, f(x), label='$f(x)$')
plt.figure(3)    # random nodes
plt.plot(x, f(x), label='$f(x)$')
plt.figure(5)    # Chebyshev nodes of first kind
plt.plot(x, f(x), label='$f(x)$')
plt.figure(7)    # Chebyshev nodes of second kind
plt.plot(x, f(x), label='$f(x)$')

# loop over node numbers
for n in nvals:

   print("Testing with n =", n)

   # uniformly-spaced nodes
   plt.figure(1)
   t = numpy.linspace(a,b,n+1)
   p = Lagrange(t, f(t), x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(2)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))

   # random nodes
   plt.figure(3)
   t = a + (b-a)*numpy.random.rand(n+1)
   p = Lagrange(t, f(t), x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(4)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))

   # Chebyshev nodes of the first kind
   plt.figure(5)
   t = (a+b)/2 + (b-a)/2*numpy.cos((2*numpy.linspace(0,n,n+1)+1)/(2*n+2)*numpy.pi)
   p = Lagrange(t, f(t), x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(6)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))

   # Chebyshev nodes of the second kind
   plt.figure(7)
   t = (a+b)/2 - (b-a)/2*numpy.cos(numpy.linspace(0,n,n+1)*numpy.pi/n)
   p = Lagrange(t, f(t), x)
   e = numpy.abs(f(x)-p)
   plt.plot(x, p, label=('$p_{%i}(x)$, error = %.2e' % (n,numpy.linalg.norm(e,numpy.inf))))
   plt.figure(8)
   plt.semilogy(x, e, label=('$|f-p_{%i}|$' % (n)))


# finalize plots
plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Uniformly-spaced nodes')

plt.figure(2)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Uniformly-spaced node error')

plt.figure(3)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Random nodes')

plt.figure(4)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Random node error')

plt.figure(5)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Chebyshev nodes of first kind')

plt.figure(6)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Chebyshev node of first kind error')

plt.figure(7)
plt.xlabel('$x$')
plt.ylabel('$f(x)$, $p(x)$')
plt.legend()
plt.title('Chebyshev nodes of second kind')

plt.figure(8)
plt.xlabel('$x$')
plt.ylabel('$|f(x) - p(x)|$')
plt.legend()
plt.title('Chebyshev node of second kind error')

plt.show()

# end of script
