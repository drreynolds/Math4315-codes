# CubicSpline.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math4315

# imports
import numpy
from LUFactorsTri    import LUFactorsTri
from BackwardSubTri  import BackwardSubTri
from ForwardSubTri   import ForwardSubTri

def CubicSplineCoeffs(t, y):
    """
    Usage: z = CubicSplineCoeffs(t, y)

    This routine computes the coefficients of the natural interpolating cubic
    spline through the data values (t_k,y_k), k=0,...,n.

    Inputs:   t - array of interpolation knots
              y - array of interpolation values
    Outputs:  z - cubic spline coefficients
    """

    # check that dimensions of t and y match
    if (numpy.shape(t) != numpy.shape(y)):
        raise ValueError("CubicSplineCoeffs error: node and data value array inputs must have identical size")

    # get overall number of knots
    n = t.size-1

    # set knot spacing array
    h = t[1:n+1] - t[0:n]

    # set diagonal values
    d = 2.0 * ( h[0:n-1] + h[1:n] )

    # set right-hand side values
    b = 6.0/h*( y[1:n+1] - y[0:n] )
    v = b[1:n] - b[0:n-1]

    # set up tridiagonal linear system, A*z=V
    A = numpy.zeros((n+1,n+1))
    V = numpy.zeros(n+1)
    for i in range(1,n):
        A[i,i-1] = h[i-1]
        A[i,i]   = d[i-1]
        A[i,i+1] = h[i]
        V[i]     = v[i-1]

    # set up first and last rows of linear system to enforce natural boundary conditions
    A[0,0] = 1.0
    V[0] = 0.0
    A[n,n] = 1.0
    V[n] = 0.0

    # solve linear system for result (using tridiagonal solvers from homework 2)
    L, U = LUFactorsTri(A)
    z = BackwardSubTri(U, ForwardSubTri(L, V))

    return z




def CubicSplineEvaluate(t, y, z, x):
    """
    Usage: s = CubicSplineEvaluate(t, y, z, x)

    This routine evaluates the cubic spline defined by the knots, t, the data
    values, y, and the coefficients, z, at the point x.

    Inputs:   t - array of interpolation knots
              y - array of interpolation values
              z - cubic spline coefficients
              x - evaluation point(s)
    outputs:  s - value of cubic spline at point(s) x
    """

    # check input arguments
    if ((t.size != y.size) or (t.size != z.size)):
        raise ValueError("CubicSplineCoeffs error: node and data value array inputs must have identical size")

    # get overall number of knots
    n = t.size-1

    # create output
    s = numpy.zeros(numpy.shape(x))

    # evaluate spline for each entry in x
    for j in range(x.size):

        # determine spline interval for this x value
        if (x[j] < t[0]):
            i = 0
        elif (x[j] > t[n]):
            i = n-1
        else:
            for i in range(n):
                if ( (x[j] >= t[i]) and (x[j] < t[i+1]) ):
                    break

        # set subinterval width
        h = t[i+1] - t[i]

        # evaluate spline
        s[j] = ( z[i]/(6.0*h)*(t[i+1]-x[j])**3 +
                 z[i+1]/(6.0*h)*(x[j]-t[i])**3 +
                 (y[i+1]/h - z[i+1]*h/6.0)*(x[j]-t[i]) +
                 (y[i]/h - z[i]*h/6.0)*(t[i+1]-x[j]) )

    return s
