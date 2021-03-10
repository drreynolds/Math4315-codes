# Chebyshev.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def Chebyshev(x, k):
    """
    Usage: T = Chebyshev(x,k)

    Function to evaluate the kth Chebyshev polynomial at a point x.

    Inputs:   x - evaluation point(s)
              k - Chebyshev polynomial index
    Outputs:  T - value(s) of T_k(x)
    """

    # evaluate using cosine formulation
    return numpy.cos(k * numpy.arccos(x))

# end of file
