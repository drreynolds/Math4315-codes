#!/usr/bin/env python3
#
# Script to test LUFactors and LUPFactors on a variety of linear systems.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
from LUFactors  import LUFactors
from LUPFactors import LUPFactors

# set matrix sizes for tests
nvals = [50, 100, 200, 400]

# set tolerance for tests
tol = numpy.sqrt(numpy.finfo(float).eps)

# tests for diagonally-dominant matrices
for n in nvals:

  print("Testing LUFactors with diagonally-dominant matrix of dimension ", n)

  # create the matrix
  A = numpy.random.rand(n,n) + n*numpy.eye(n)

  # construct and test LU factorization
  L, U = LUFactors(A)
  checks_out = True
  if ( (numpy.linalg.norm(L - numpy.tril(L))>tol) or
       (numpy.linalg.norm(numpy.diag(L)-numpy.ones([n,1])) > tol) ):
    checks_out = False
    print("  LUFactors failure: L is not unit lower-triangular")
  if (numpy.linalg.norm(U - numpy.triu(U))>tol):
    checks_out = False
    print("  LUFactors failure: U is not upper-triangular")
  if (numpy.linalg.norm(A-L@U)>tol):
    checks_out = False
    print("  LUFactors failure: A != LU")
  if (checks_out):
    print("  LUFactors passes all tests")


# tests for nonsingular (but non-diagonally-dominant) matrices
for n in nvals:

  print("Testing LUPFactors with non-diagonally-dominant matrix of dimension ", n)

  # create the matrix
  A = numpy.random.rand(n,n)
  for i in range(n):
    A[i,n-1-i] += n

  # construct and test LUP factorization
  L, U, P = LUPFactors(A)
  checks_out = True
  if ( (numpy.linalg.norm(L - numpy.tril(L))>tol) or
       (numpy.linalg.norm(numpy.diag(L)-numpy.ones([n,1])) > tol) ):
    checks_out = False
    print("  LUPFactors failure: L is not unit lower-triangular")
  if (numpy.linalg.norm(U - numpy.triu(U))>tol):
    checks_out = False
    print("  LUPFactors failure: U is not upper-triangular")
  if (numpy.linalg.norm(P.T@P - numpy.eye(n))>tol):
    checks_out = False
    print("  LUPFactors failure: P is not a permutation matrix")
  if (numpy.linalg.norm(A-P.T@L@U)>tol):
    checks_out = False
    print("  LUPFactors failure: A ~= P^T L U")
  if (checks_out):
    print("  LUPFactors passes all tests")


# ensure that singular case fails
n = 100
print("Testing with singular matrices (should fail)")
A = numpy.random.rand(n,n)
A[:,n-21] = A[:,0] - 4*A[:,9]
try:
  L, U = LUFactors(A)
except:
  print("   singularity correctly caught by LUFactors")
try:
  L, U, P = LUPFactors(A)
except:
  print("   singularity correctly caught by LUPFactors")

# end of script
