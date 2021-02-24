# LUFactorsTri.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def LUFactorsTri(T):
    """
    usage: [L,U] = LUFactorsTri(T)

    Row-oriented naive LU factorization for tridiagonal matrices; constructs the 
    factorization
          T = LU
    This function checks that T is square, and attempts to catch the case where T
    is singular.

    Inputs:
      T - tridiagonal n-by-n matrix

    Outputs:
      L - unit-lower-triangular 'tridiagonal' matrix (n-by-n)
      U - upper-triangular 'tridiagonal' matrix (n-by-n)
    """

    # check input
    m, n = numpy.shape(T)
    if (m != n):
        raise ValueError("LUFactorsTri error: matrix must be square")

    # set singularity tolerance
    tol = 1000*numpy.finfo(float).eps

    # create output matrices
    U = T.copy()
    L = numpy.eye(n)
    for k in range(n-1):             # loop over pivots
      if (numpy.abs(U[k,k]) < tol):  # check for singularity
        raise ValueError("LUFactors error: factorization failure")
      L[k+1,k] = U[k+1,k]/U[k,k]     # compute multiplier for next row
      for j in range(k,k+2):         # update remainder of matrix row
        U[k+1,j] -= L[k+1,k]*U[k,j]

    return [L, U]

# end function
