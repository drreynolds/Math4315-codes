# newton.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

def newton(Ffun, Jfun, x, maxit, rtol, atol, output):
    """
    Usage: x, its = newton(Ffun, Jfun, x, maxit, rtol, atol, output)

    This routine uses Newton's method to approximate a root of
    the nonlinear system of equations F(x)=0.  The iteration ceases
    when the following condition is met:

       ||xnew - xold|| < atol + rtol*||xnew||

    inputs:   Ffun     nonlinear residual function
              Jfun     Jacobian function
              x        initial guess at solution
              maxit    maximum allowed number of iterations
              rtol     relative solution tolerance
              atol     absolute solution tolerance
              output   flag (true/false) to output iteration history/plot
    outputs:  x        approximate solution
              its      number of iterations taken
    """

    # imports
    import numpy

    # check input arguments
    if (int(maxit) < 1):
        print("newton: maxit = %i < 1. Resetting to 10" % (int(maxit)))
        maxit = 10
    if (rtol < 1e-15):
        print("newton: rtol = %g < %g. Resetting to %g" % (rtol, 1e-15, 1e-15))
        rtol = 1e-15
    if (atol < 0):
        print("newton: atol = %g < 0. Resetting to %g" % (atol, 1e-15))
        atol = 1e-15

    # evaluate initial residual
    F = Ffun(x)

    # perform iteration
    for its in range(1,maxit+1):

        # evaluate derivative
        J = Jfun(x)

        # compute Newton update, new guess at solution, new residual
        if (numpy.isscalar(x)):    # if problem is scalar-valued
            h = F/J
        else:                      # if problem is vector-valued
            h = numpy.linalg.solve(J, F)
        x = x - h
        F = Ffun(x)

        # check for convergence and output diagnostics
        hnorm = numpy.linalg.norm(h)
        xnorm = numpy.linalg.norm(x)
        if (output):
            print("   iter %3i, \t||h|| = %g, \ttol = %g" % (its, hnorm, atol + rtol*xnorm))

        if (hnorm < atol + rtol*xnorm):
            break

    return [x, its]
