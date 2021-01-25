# LUFactors.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def LUFactors(A):
    """
    usage: [L,U] = LUFactors(A)

    Row-oriented naive LU factorization; constructs the factorization
          A = LU
    This function checks that A is square, and attempts to catch the case where A
    is singular.

    Inputs:
      A - square n-by-n matrix

    Outputs:
      L - unit-lower-triangular matrix (n-by-n)
      U - upper-triangular matrix (n-by-n)
    """

    # check input
    m, n = numpy.shape(A)
    if (m != n):
        raise ValueError("LUFactors error: matrix must be square")

    # set singularity tolerance
    tol = 1000*numpy.finfo(float).eps

    # create output matrices
    U = A.copy()
    L = numpy.eye(n)
    for k in range(n-1):             # loop over pivots
      if (numpy.abs(U[k,k]) < tol):  # check for singularity
        raise ValueError("LUFactors error: factorization failure")
      for i in range(k+1,n):         # loop over remaining rows
        L[i,k] = U[i,k]/U[k,k]       # compute multiplier
        for j in range(k,n):         # update remainder of matrix row
          U[i,j] -= L[i,k]*U[k,j]

    return [L, U]

# end function
