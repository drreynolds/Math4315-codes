# ivp_methods.py
#
# Collection of single-step initial-value problem methods.
# Each of these apply a specific one-step method to
# approximate one time step of an IVP solver
#    u_{n+1} = u_n + h*phi(t_n, u_n, h; f)
#
# All of these require that the ODE RHS function f have
# the calling syntax
#    u' = f(t,u)
# and that it return u' of the same type (scalar vs numpy array)
# and shape as u (in case it is array-valued)
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
from newton import newton


### Explicit methods ###

def fwd_Euler_step(f, told, uold, h):
    """
    Usage: unew = fwd_Euler_step(f, told, uold, h)

    Forward Euler solver for one step of the ODE problem,
       u' = f(t,u), t in tspan,
       u(t0) = u0.

    Inputs:  f = function for ODE right-hand side, f(t,u)
             told = current time
             uold = current solution
             h = time step size
    Outputs: unew = updated solution
    """
    return (uold + h*f(told, uold))


def RK2_step(f, told, uold, h):
    """
    Usage: unew = RK2_step(f, told, uold, h)

    Runge-Kutta of order 2 solver for one step of the ODE
    problem,
       u' = f(t,u), t in tspan,
       u(t0) = u0.

    Inputs:  f = function for ODE right-hand side, f(t,u)
             told = current time
             uold = current solution
             h = time step size
    Outputs: unew = updated solution
    """
    # get ODE RHS at this time step
    f1 = f(told, uold)

    # set stages z1 and z2
    z1 = uold
    z2 = uold + h/2*f1

    # get f(t+h/2,z2)
    f2 = f(told+h/2, z2)

    # update solution in time
    return (uold + h*f2)


def Heun_step(f, told, uold, h):
    """
    Usage: unew = Heun_step(f, told, uold, h)

    Heun's Runge-Kutta of order 2 solver for one step of the
    ODE problem,
       u' = f(t,u), t in tspan,
       u(t0) = u0.

    Inputs:  f = function for ODE right-hand side, f(t,u)
             told = current time
             uold = current solution
             h = time step size
    Outputs: unew = updated solution
    """
    # call f to get ODE RHS at this time step
    f1 = f(told, uold)

    # set stages z1 and z2
    z1 = uold
    z2 = uold + h*f1

    # get f(t+h,z2)
    f2 = f(told+h, z2)

    # update solution in time
    return (uold + h/2*(f1+f2))


def AB2_step(f, tcur, ucur, h, fold):
    """
    Usage: unew, fcur = AB2_step(f, tcur, ucur, h, fold)

    Adams-Bashforth method of order 2 for one step of the ODE
    problem,
       u' = f(t,u), t in tspan,
       u(t0) = u0.

    Inputs:  f = function for ODE right-hand side, f(t,u)
             tcur = current time
             ucur = current solution
             h = time step size
             fold = RHS evaluated at previous time step
    Outputs: unew = updated solution
             fcur = RHS evaluated at current time step
    """
    # get ODE RHS at this time step
    fcur = f(tcur, ucur)

    # update solution in time, and return
    unew = ucur + h/2*(3*fcur - fold)
    return [unew, fcur]



### Implicit methods ###

def bwd_Euler_step(f, f_u, told, uold, h):
    """
    Usage: unew = bwd_Euler_step(f, f_u, told, uold, h)

    Backward Euler solver for one step of the ODE problem,
       u' = f(t,u), t in tspan,
       u(t0) = u0.

    Inputs:  f = function for ODE right-hand side, f(t,u)
             f_u = function for ODE Jacobian, f_u(t,u)
             told = current time
             uold = current solution
             h = time step size
    Outputs: unew = updated solution
    """

    # set nonlinear solver parameters
    maxit = 20
    rtol = 1e-9
    atol = 1e-12
    output = False

    # create implicit residual and Jacobian functions
    if (numpy.isscalar(uold)):     # if problem is scalar-valued
        I = 1.0
    else:                          # if problem is vector-valued
        I = numpy.identity(uold.size)
    def F(unew):
        return (unew - uold - h*f(told+h,unew))
    def A(unew):
        return (I - h*f_u(told+h,unew))

    # perform implicit solve
    unew, its = newton(F, A, uold, maxit, rtol, atol, output)
    return unew


def trap_step(f, f_u, told, uold, h):
    """
    Usage: unew = trap_step(f, f_u, told, uold, h)

    Trapezoidal (2nd-order Adams-Moulton) solver for one step of the ODE problem,
       u' = f(t,u), t in tspan,
       u(t0) = uy.

    Inputs:  f = function for ODE right-hand side, f(t,u)
             f_u = function for ODE Jacobian, f_u(t,u)
             told = current time
             uold = current solution
             h = time step size
    Outputs: unew = updated solution
    """

    # set nonlinear solver parameters
    maxit = 20
    rtol = 1e-9
    atol = 1e-12
    output = False

    # create implicit residual and Jacobian functions
    fold = f(told, uold)
    if (numpy.isscalar(uold)):     # if problem is scalar-valued
        I = 1.0
    else:                          # if problem is vector-valued
        I = numpy.identity(uold.size)
    def F(unew):
        return (unew - uold - 0.5*h*(f(told+h,unew)+fold))
    def A(unew):
        return (I - 0.5*h*f_u(told+h,unew))

    # perform implicit solve
    unew, its = newton(F, A, uold, maxit, rtol, atol, output)
    return unew


# end of functions
