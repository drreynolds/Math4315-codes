# LinearSpline.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def LinearSpline(x,t,y):
    """
    usage: p = LinearSpline(x,t,y)

    Function to evaluate the linear spline defined by the data values
    (t_k,y_k), k=0,...,n, at the point(s) x.

    Inputs:   x - point(s) to evaluate linear spline
              t - array of interpolation nodes
              y - array of interpolation values
    Outputs:  p - value of linear spline at point(s) x
    """

    # check that dimensions of t and y match
    if (numpy.shape(t) != numpy.shape(y)):
        raise ValueError("LinearSpline error: node and data value array inputs must have identical size")

    # get overall number of nodes
    n = numpy.size(t)-1

    # initialize output
    p = numpy.zeros(numpy.size(x))

    # evaluate p by adding in contributions from each hat function

    #   left-most hat function
    p += y[0]*numpy.maximum(0,(t[1]-x)/(t[1]-t[0]))

    #   right-most hat function
    if (n > 0):
        p += y[n]*numpy.maximum(0,(x-t[n-1])/(t[n]-t[n-1]))

    # intermediate hat functions
    for k in range(1,n):
        p += y[k]*numpy.maximum(0,numpy.minimum((x-t[k-1])/(t[k]-t[k-1]),(t[k+1]-x)/(t[k+1]-t[k])))

    return p

# end function
