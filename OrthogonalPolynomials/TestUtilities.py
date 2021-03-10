#!/usr/bin/env python3
#
# Script to verify polynomial bases and inner product routine.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
from Chebyshev import Chebyshev
from Legendre import Legendre
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

# quick test to verify orthogonality of first 41 Chebyshev polynomials
print('Testing orthogonality of first 41 Chebyshev polynomials:')
passed = True
for i in range(41):
    for j in range(i+1,41):
        def fi(x):
            return Chebyshev(x,i)
        def fj(x):
            return Chebyshev(x,j)
        v = L2InnerProduct(fi,fj,wC,a,b)
        if (abs(v) > 1e-6):
            print('  <p%i,p%i> = %e' % (i, j, v))
            passed = False
if (passed):
    print('  Tests passed')

# quick test to verify orthogonality of first 41 Legendre polynomials
print('Testing orthogonality of first 41 Legendre polynomials:')
passed = True
for i in range(41):
    for j in range(i+1,41):
        def fi(x):
            return Legendre(x,i)
        def fj(x):
            return Legendre(x,j)
        v = L2InnerProduct(fi,fj,wL,a,b)
        if (abs(v) > 1e-6):
            print('  <p%i,p%i> = %e' % (i, j, v))
            passed = False
if (passed):
    print('  Tests passed')

# quick test to verify norms for first 41 Chebyshev polynomials
print('Testing norms for first 41 Chebyshev polynomials:')
passed = True
for i in range(41):
    def fi(x):
        return Chebyshev(x,i)
    v = L2InnerProduct(fi,fi,wC,a,b)
    if (i==0):
        if (abs(v - numpy.pi) > 1e-6):
            print('  <p0,p0> = %e' % (v))
            passed = False
    else:
        if (abs(v - numpy.pi/2) > 1e-6):
            print('  <p%i,p%i> = %e' % (i, i, v))
            passed = False
if (passed):
    print('  Tests passed')

# quick test to verify norms for first 41 Legendre polynomials
print('Testing norms for first 41 Legendre polynomials:')
passed = True
for i in range(41):
    def fi(x):
        return Legendre(x,i)
    v = L2InnerProduct(fi,fi,wL,a,b)
    if (abs(v - 1.0/(i+0.5)) > 1e-6):
        print('  <p0,p0> = %e' % (v))
        passed = False
if (passed):
    print('  Tests passed')

# end of script
