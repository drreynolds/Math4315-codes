# differentiation_matrix.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

def differentiation_matrix(n, a, b, order, deriv):
    """
    Usage: x, D = differentiation_matrix(n, a, b, order, deriv)

    Utility to compute differentiation matrix of specified order
    of accuracy and derivative order, over an interval [a,b].

    Inputs:  n = number of intervals to use
             a,b = interval to discretize
             order = differentiation matrix type:
                      0 = Chebyshev [spectral convergence]
                      1 = O(h) finite-difference over regular mesh
                      2 = O(h^2) finite-difference over regular mesh
             deriv = derivative order {1,2}
    Outputs: x = column vector containing partition of [a,b]
             D = differentiation matrix
    """

    # imports
    import numpy

    # check for a sufficient number of intervals
    if (n < 2):
        raise ValueError("insufficient number of intervals")

    # ensure that n, order and deriv are integers
    n = int(n)
    order = int(order)
    deriv = int(deriv)

    # check for valid order
    if ((order < 0) or (order > 2)):
        raise ValueError("invalid order selected")

    # construct matrix based on desired differentiation order

    if (order == 0):  # Chebyshev [based off of 'diffcheb' function 10.2.2 from the book]

        # set Chebyshev nodes in [-1,1]
        x = -numpy.cos( numpy.linspace(0,n,n+1)*numpy.pi/n )

        # create base differentiation matrix
        Dbase = numpy.zeros((n+1,n+1), dtype=float)
        c = numpy.ones((n+1), dtype=float)
        c[0] = 2
        c[-1] = 2
        i = numpy.linspace(0,n,n+1,dtype=int)
        for j in range(n+1):
            num = c[i]*(-1)**(i+j)
            den = c[j]*(x - x[j])
            for k in range(j):
                Dbase[k,j] = num[k]/den[k]
            for k in range(j+1,n+1):
                Dbase[k,j] = num[k]/den[k]
        Dbase = Dbase - numpy.diag(numpy.sum(Dbase,1))

        # remap to interval [a,b]
        x = a + (b-a)/2*(x+1)
        Dbase = (2/(b-a))*Dbase

        # construct output matrix through multiplication
        D = numpy.copy(Dbase)
        for i in range(2,deriv+1):
            D = D @ Dbase

    else:             # finite-difference

        # set uniform nodes and corresponding h
        x = numpy.linspace(a,b,n+1)
        h = (b-a)/n

        # first order, first derivative
        if ((order == 1) and (deriv == 1)):

            D = numpy.diag(numpy.ones(n),1) - numpy.diag(numpy.ones(n+1), 0)
            D[n,n-1:n+1] = numpy.array([-1, 1])
            D *= (1/h)

        # second order, first derivative
        elif ((order == 2) and (deriv == 1)):

            D = 0.5*(numpy.diag(numpy.ones(n),1) - numpy.diag(numpy.ones(n),-1))
            D[0,0:3] = numpy.array([-1.5, 2.0, -0.5])
            D[n,n-2:n+1] = numpy.array([0.5, -2.0, 1.5])
            D *= (1/h)

        # second order, second derivative
        elif ((order == 2) and (deriv == 2)):

            D = numpy.diag(numpy.ones(n),1) + numpy.diag(numpy.ones(n),-1) - 2*numpy.diag(numpy.ones(n+1),0)
            D[0,0:4] = numpy.array([2.0, -5.0, 4.0, -1.0])
            D[n,n-3:n+1] = numpy.array([-1.0, 4.0, -5.0, 2.0])
            D *= (1/h/h)

        # all other choices are not implemented
        else:
            raise ValueError("invalid order/deriv selection for finite-difference matrix")

    return [x, D]


# end of file
