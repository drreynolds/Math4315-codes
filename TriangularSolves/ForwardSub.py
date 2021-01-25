# ForwardSub.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def ForwardSub(L,b):
    """
    usage: y = ForwardSub(L,b)

    Row-oriented forward substitution to solve the lower-triangular
    linear system
          L y = b
    This function does not ensure that L is lower-triangular.  It does,
    however, attempt to catch the case where L is singular.

    Inputs:
      L - square n-by-n matrix (assumed lower triangular)
      b - right-hand side vector (n-by-1)

    Outputs:
      y - solution vector (n-by-1)
    """

    # check inputs
    m, n = numpy.shape(L)
    if (m != n):
        raise ValueError("ForwardSub error: matrix must be square")
    p = numpy.size(b)
    if (p != n):
        raise ValueError("ForwardSub error: right-hand side vector has incorrect dimensions")
    if (numpy.min(numpy.abs(numpy.diag(L))) < 100*numpy.finfo(float).eps):
        raise ValueError("ForwardSub error: matrix is [close to] singular")

    # create output vector
    y = b.copy()

    # perform forward-subsitution algorithm
    for i in range(n):
        for j in range(i):
            y[i] -= L[i,j]*y[j]
        y[i] /= L[i,i]

    return y

# end function
