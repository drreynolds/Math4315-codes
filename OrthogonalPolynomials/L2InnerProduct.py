# L2InnerProduct.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy


# custom, high-accuracy, adaptive numerical integration utility
# (since none seem to be built into numpy)
def Gauss8(f, a, b):
    """
    Usage: I = Gauss8(f, a, b)

    Function to perform O((b-a)^16) Gaussian quadrature of a
    function f over the interval [a,b].
    """
    nd = (a+b)/2 + (b-a)/2*numpy.array([-0.18343464249564980493, 0.18343464249564980493,
                                        -0.52553240991632898581, 0.52553240991632898581,
                                        -0.79666647741362673959, 0.79666647741362673959,
                                        -0.96028985649753623168, 0.96028985649753623168])
    wt = (b-a)/2.0*numpy.array([0.36268378337836198296, 0.36268378337836198296,
                                0.31370664587788728733, 0.31370664587788728733,
                                0.22238103445337447054, 0.22238103445337447054,
                                0.10122853629037625915, 0.10122853629037625915])
    return numpy.sum(wt * f(nd))

def AdaptiveInt(f, a=-1, b=1, rtol=1e-5, atol=1e-9):
    """
    Usage: I = AdaptiveInt(f, a, b, rtol, atol)

    Function to adaptively compute the integral of f over the interval
    [a,b] to a tolerance of rtol*|I| + atol.

    The input f is required; all other inputs are optional, with default
    values [a,b] = [-1,1], rtol=1e-5 and atol=1e-9.
    """

    # if interval is too narrow, return with current approximation
    m = (a+b)/2
    if ((m-a) < numpy.finfo(float).eps*(abs(a)+abs(b))):
        return Gauss8(f, a, b)

    # compute overall quadrature and left/right quadratures
    I0 = Gauss8(f, a, b)
    I1 = Gauss8(f, a, m)
    I2 = Gauss8(f, m, b)

    # return with better approximation if error is sufficiently small
    if ( abs(I1+I2-I0) < rtol*abs(I1+I2)+atol ):
        return I1+I2

    # call AdaptiveInt separately on both halves (recursion), and return with sum
    return (AdaptiveInt(f, a, m, rtol, atol) + AdaptiveInt(f, m, b, rtol, atol))



# actual routine to be used in demonstration
def L2InnerProduct(f, g, w, a=-1, b=1):
    """
    Usage: v = L2InnerProduct(f, g, w, a, b)

    Function to evaluate the weighted L^2 inner product between two functions,
    f and g, over an interval [a,b], based on the weight function w(x).

    Inputs:   f - function handle
              g - function handle
              w - function handle (assumed to have strictly positive values)
              a - left endpoint of interval (default = -1)
              b - right endpoint of interval (default = 1)
    Outputs:  v - value of inner product
    """

    # ensure that interval is valid
    if ((b - a) < 2*numpy.finfo(float).eps):
        raise ValueError("L2InnerProduct error: invalid interval")

    # set integrand
    def integrand(x):
        return f(x)*g(x)*w(x)

    # approximate integral over [a,b]
    try:
        import scipy.integrate as integrate
        v, err = integrate.quad(integrand, a, b)
        return v
    except ImportError:
        return AdaptiveInt(integrand, a+numpy.finfo(float).eps,
                           b-numpy.finfo(float).eps, 1e-8, 1e-11)


# end of file
