# Lagrange.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def Lagrange(t, y, x):
    """
    Usage: p = Lagrange(t,y,x)

    Function to evaluate the Lagrange interpolating polynomial p(x)
    defined in the Lagrange basis by
        p(x) = y(1)*l1(x) + y(2)*l2(x) + ... + y(n)*ln(x).

    Inputs:   t - array of interpolation nodes
              y - array of interpolation values
              x - point(s) to evaluate Lagrange interpolant
    Outputs:  p - value of Lagrange interpolant at point(s) x
    """

    # check inputs
    if (t.size != y.size):
        raise ValueError("Lagrange error: (t,y) have different sizes")

    n = t.size

    # initialize output
    p = numpy.zeros(x.shape)

    # iterate over Lagrange basis functions
    for k in range(n):

        # initialize l (the kth Lagrange basis function)
        l = numpy.ones(x.shape)

        # iterate over data to construct l(x)
        for j in range(n):
            # exclude the k-th data point
            if (j != k):
                l *= (x-t[j]) / (t[k]-t[j])

        # add contribution from this basis function (and data value) into p
        p += y[k]*l

    return p

# end of file
