# BackwardSub.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def BackwardSub(U,y):
    """
    usage: x = BackwardSub(U,y)

    Row-oriented backward substitution to solve the upper-triangular
    linear system
          U x = y
    This function does not ensure that U is upper-triangular.  It does,
    however, attempt to catch the case where U is singular.

    Inputs:
      U - square n-by-n matrix (assumed upper triangular)
      y - right-hand side vector (n-by-1)

    Outputs:
      x - solution vector (n-by-1)
    """

    # check inputs
    m, n = numpy.shape(U)
    if (m != n):
        raise ValueError("BackwardSub error: matrix must be square")
    p = numpy.size(y)
    if (p != n):
        raise ValueError("BackwardSub error: right-hand side vector has incorrect dimensions")
    if (numpy.min(numpy.abs(numpy.diag(U))) < 100*numpy.finfo(float).eps):
        raise ValueError("BackwardSub error: matrix is [close to] singular")

    # create output vector
    x = y.copy()

    # perform forward-subsitution algorithm
    for i in range(n-1,-1,-1):
        for j in range(i+1,n):
            x[i] -= U[i,j]*x[j]
        x[i] /= U[i,i]

    return x

# end function
