#!/usr/bin/env python3
#
# Script to demonstrate effects of matrix conditioning in floating-point arithmetic.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

# utility function to create Hilbert matrix of dimension n
def Hilbert(n):
  H = numpy.zeros([n,n])
  for i in range(n):
    for j in range(n):
      H[i,j] = 1/(1+i+j)
  return H


# set matrix sizes for tests
nvals = [6, 8, 10, 12, 14]

# run tests for each matrix size
for n in nvals:

  # create matrix, solution and right-hand side vector
  A = Hilbert(n)
  x = numpy.random.rand(n,1)
  b = A@x

  # ouptut condition number
  print("Hilbert matrix of dimension", n, ": condition number = ",
        format(numpy.linalg.cond(A),'e'))

  # solve the linear system
  x_comp = numpy.linalg.solve(A,b)

  # output relative solution error
  print("  relative solution error = ",
        numpy.linalg.norm(x-x_comp)/numpy.linalg.norm(x))

# end of script
