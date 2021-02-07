# LUPPFactors.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def LUPPFactors(A):
    """
    usage: [L,U,P1,P2] = LUPPFactors(A)

    Row-oriented LU factorization with complete pivoting; constructs the factorization
          P1 A P2 = LU  <=>  A = P1^T L U P2^T
    This function checks that A is square, and attempts to catch the case where A
    is singular.

    Inputs:
      A - square n-by-n matrix

    Outputs:
      L  - unit-lower-triangular matrix (n-by-n)
      U  - upper-triangular matrix (n-by-n)
      P1 - permutation matrix (n-by-n)
      P2 - permutation matrix (n-by-n)
    """

    # check input
    m, n = numpy.shape(A)
    if (m != n):
        raise ValueError("LUPPFactors error: matrix must be square")

    # set singularity tolerance
    tol = 1000*numpy.finfo(float).eps

    # create output matrices
    U = A.copy()
    L = numpy.eye(n)
    P1 = numpy.eye(n)
    P2 = numpy.eye(n)
    for k in range(n-1):               # loop over pivots
      si = k                           # determine pivot position
      sj = k
      for i in range(k,n):
        for j in range(k,n):
          if (numpy.abs(U[i,j]) > numpy.abs(U[si,sj])):
            si = i
            sj = j
      if (numpy.abs(U[si,sj]) < tol):  # check for singularity
        raise ValueError("LUPPFactors error: matrix is [close to] singular")
      for j in range(k,n):             # swap rows in U
        tmp = U[k,j]
        U[k,j] = U[si,j]
        U[si,j] = tmp
      for j in range(k):               # swap rows in L
        tmp = L[k,j]
        L[k,j] = L[si,j]
        L[si,j] = tmp
      for j in range(n):               # swap rows in P1
        tmp = P1[k,j]
        P1[k,j] = P1[si,j]
        P1[si,j] = tmp
      for i in range(n):               # swap columns in U
        tmp = U[i,k]
        U[i,k] = U[i,sj]
        U[i,sj] = tmp
      for i in range(n):               # swap columns in P2
        tmp = P2[i,k]
        P2[i,k] = P2[i,sj]
        P2[i,sj] = tmp
      for i in range(k+1,n):           # loop over remaining rows
        L[i,k] = U[i,k]/U[k,k]         # compute multiplier
        for j in range(k,n):           # update remainder of matrix row
          U[i,j] -= L[i,k]*U[k,j]

    return [L, U, P1, P2]

# end function
