#!/usr/bin/env python3
#
# Script to test ForwardSub and BackwardSub on a variety of linear systems.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
from ForwardSub  import ForwardSub
from BackwardSub import BackwardSub

# set matrix sizes for tests
nvals = [50, 100, 200, 400]

# full-rank square matrix tests
for n in nvals:

    print("Testing with full-rank triangular matrices of dimension ", n)

    # create the matrices
    L = numpy.tril(numpy.random.rand(n,n) + 2*numpy.eye(n))
    U = numpy.triu(numpy.random.rand(n,n) + 2*numpy.eye(n))

    # create solution vector
    x = numpy.random.rand(n)

    # solve triangular linear systems
    x_bs = BackwardSub(U, U@x)
    x_fs = ForwardSub(L, L@x)

    # output results
    print("   BackwardSub error = ", numpy.linalg.norm(x-x_bs))
    print("   ForwardSub error  = ", numpy.linalg.norm(x-x_fs))

# ensure that rank-deficient case fails
n = 100
print("Testing with rank-deficient triangular matrices (should fail)")
L = numpy.tril(numpy.random.rand(n,n))
L[n-4,n-4] = numpy.finfo(float).eps
U = numpy.triu(numpy.random.rand(n,n))
U[n-6,n-6] = numpy.finfo(float).eps
x = numpy.random.rand(n)
try:
  x_bs = BackwardSub(U, U@x)
except:
  print("   rank deficiency correctly caught by BackwardSub")
try:
  x_fs = ForwardSub(L, L@x)
except:
  print("   rank deficiency correctly caught by ForwardSub")

# end of script
