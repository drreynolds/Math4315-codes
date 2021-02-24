# ForwardSubTri.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def ForwardSubTri(L,b):
    """
    usage: y = ForwardSub(L,b)

    Row-oriented forward substitution to solve the lower-triangular, 'tridiagonal'
    linear system
          L y = b
    This function does not ensure that L has the correct nonzero structure.  It does,
    however, attempt to catch the case where L is singular.

    Inputs:
      L - square n-by-n matrix (assumed lower triangular and 'tridiagonal')
      b - right-hand side vector (n-by-1)

    Outputs:
      y - solution vector (n-by-1)
    """

    # check inputs
    m, n = numpy.shape(L)
    if (m != n):
        raise ValueError("ForwardSubTri error: matrix must be square")
    p = numpy.size(b)
    if (p != n):
        raise ValueError("ForwardSubTri error: right-hand side vector has incorrect dimensions")
    if (numpy.min(numpy.abs(numpy.diag(L))) < 100*numpy.finfo(float).eps):
        raise ValueError("ForwardSubTri error: matrix is [close to] singular")

    # create output vector
    y = b.copy()

    # perform tridiagonal forward-subsitution algorithm
    for i in range(n):
        if (i>0):
            y[i] -= L[i,i-1]*y[i-1]
        y[i] /= L[i,i]

    return y

# end function
