#!/usr/bin/env python3
#
# Script to test LUPFactors_simple, LUPFactors and LUPPFactors on a variety
# of ill-conditioned matrices.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
from LUPFactors_simple import LUPFactors_simple
from LUPFactors import LUPFactors
from LUPPFactors import LUPPFactors

# set matrix sizes for tests
nvals = [20, 40, 80, 160]

# loop over matrix sizes
for n in nvals:

  print("Testing stabilization approaches for linear system of dimension ", n)

  # create the matrix
  A = numpy.vander(numpy.linspace(0.1,1,n)) + 0.0001*numpy.random.rand(n,n)
  n2 = n//2
  randrows = numpy.random.randint(0,n-1,n2)
  randcols = numpy.random.randint(0,n-1,n2)
  A[randrows,:] = numpy.diag(1000*numpy.random.rand(n2))@A[randrows,:]
  A[:,randcols] = A[:,randcols]@numpy.diag(1000*numpy.random.rand(n2))

  # test LUPFactors_simple
  print("  LUPFactors_simple:")
  L, U, P = LUPFactors_simple(A)
  print("    norm(A - P^T L U) = ", numpy.linalg.norm(A - P.T@L@U))

  # test LUPFactors
  print("  LUPFactors:")
  L, U, P = LUPFactors(A)
  print("    norm(A - P^T L U) = ", numpy.linalg.norm(A - P.T@L@U))

  # test LUPPFactors
  print("  LUPPFactors:")
  L, U, P1, P2 = LUPPFactors(A)
  print("    norm(A - P1^T L U P2^T) = ", numpy.linalg.norm(A - P1.T@L@U@P2.T))

# end of script
