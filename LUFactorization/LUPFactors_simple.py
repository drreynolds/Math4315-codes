# LUPFactors_simple.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def LUPFactors_simple(A):
    """
    usage: [L,U,P] = LUPFactors_simple(A)

    Row-oriented LU factorization with simplistic row swapping; constructs the
    factorization
          P A = LU  <=>  A = P^T L U
    This function checks that A is square, and attempts to catch the case where A
    is singular.

    Inputs:
      A - square n-by-n matrix

    Outputs:
      L - unit-lower-triangular matrix (n-by-n)
      U - upper-triangular matrix (n-by-n)
      P - permutation matrix (n-by-n)
    """

    # check input
    m, n = numpy.shape(A)
    if (m != n):
        raise ValueError("LUPFactors_simple error: matrix must be square")

    # set singularity tolerance
    tol = 1000*numpy.finfo(float).eps

    # create output matrices
    U = A.copy()
    L = numpy.eye(n)
    P = numpy.eye(n)
    for k in range(n-1):             # loop over pivots
      if (numpy.abs(U[k,k]) < tol):  # check for 'zero' pivot
        s = k                        # find first 'nonzero' pivot below
        for i in range(k+1,n):
          if (numpy.abs(U[i,k]) > tol):
            s = i
            break
        if (numpy.abs(U[s,k]) < tol):  # check for singularity
          raise ValueError("LUPFactors_simple error: matrix is [close to] singular")
        for j in range(k,n):           # swap rows in U
          tmp = U[k,j]
          U[k,j] = U[s,j]
          U[s,j] = tmp
        for j in range(k):             # swap rows in L
          tmp = L[k,j]
          L[k,j] = L[s,j]
          L[s,j] = tmp
        for j in range(n):             # swap rows in P
          tmp = P[k,j]
          P[k,j] = P[s,j]
          P[s,j] = tmp
      for i in range(k+1,n):         # loop over remaining rows
        L[i,k] = U[i,k]/U[k,k]       # compute multiplier
        for j in range(k,n):         # update remainder of matrix row
          U[i,j] -= L[i,k]*U[k,j]

    return [L, U, P]

# end function
