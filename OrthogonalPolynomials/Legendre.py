# Legendre.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def Legendre(x, k):
    """
    Usage: p = Legendre(x,k)

    Function to evaluate the kth Legendre polynomial at a point x.

    Inputs:   x - evaluation point(s)
              k - Legendre polynomial index
    Outputs:  p - value(s) of p_k(x)
    """

    # quick return for first two Legendre polynomials
    if (k == 0):
        if (numpy.isscalar(x)):       # if problem is scalar-valued
            return 1.0
        else:
            return numpy.ones(x.shape)
    if (k == 1):
        if (numpy.isscalar(x)):
            return x
        else:
            return x.copy()

    # initialize 3-term recurrence
    if (numpy.isscalar(x)):
        p0 = 1.0
        p1 = x
    else:
        p0 = numpy.ones(x.shape)
        p1 = x.copy()

    # perform recurrence to evaluate p, and update 'old' values
    for i in range(2,k+1):
        p = (2.0*i-1.0)/i*x*p1 - (i-1.0)/i*p0
        if (numpy.isscalar(x)):
            p0 = p1
            p1 = p
        else:
            p0 = p1.copy()
            p1 = p.copy()

    return p

# end of file
