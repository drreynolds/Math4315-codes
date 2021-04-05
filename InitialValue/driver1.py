#!/usr/bin/env python3
#
# Script to test various ODE solvers for the test problem
#    u' = (u+t^2-2)/(t+1),   t in [0,5]
#    u(0) = 2.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import matplotlib.pyplot as plt
from ivp_methods import *

# ODE RHS and Jacobian, initial condition, etc.
def f(t,u):
    return (u+t*t-2)/(t+1)
def f_u(t,u):
    return (1.0/(t+1))
def utrue(t):
    return (t**2 + 2*t + 2 - 2*(t+1)*numpy.log(t+1))
u0  = 2.0
t0  = 0.0
Tf  = 5.0
hvals = [0.5**2, 0.5**3, 0.5**4, 0.5**5, 0.5**6, 0.5**7, 0.5**8]

# initialize error storage arrays
err_fE     = numpy.zeros((len(hvals),1), dtype=float)
err_RK2    = numpy.zeros((len(hvals),1), dtype=float)
err_Heun   = numpy.zeros((len(hvals),1), dtype=float)
err_AB2    = numpy.zeros((len(hvals),1), dtype=float)
err_bE     = numpy.zeros((len(hvals),1), dtype=float)
err_trap   = numpy.zeros((len(hvals),1), dtype=float)

# iterate over our h values
for j, h in enumerate(hvals):

    # create time output array
    N = int((Tf-t0)/h)
    tvals = numpy.linspace(t0, Tf, N+1)

    # initialize solutions
    u_fE     = u0
    u_RK2    = u0
    u_Heun   = u0
    u_AB2    = u0
    u_bE     = u0
    u_trap   = u0

    # try out our methods
    for i in range(N):

        t = tvals[i]
        tnew = t + h

        u_fE = fwd_Euler_step(f, t, u_fE,  h)
        err_fE[j] = max(err_fE[j], abs(u_fE-utrue(tnew)))

        u_RK2 = RK2_step(f, t, u_RK2, h)
        err_RK2[j] = max(err_RK2[j], abs(u_RK2-utrue(tnew)))

        u_Heun = Heun_step(f, t, u_Heun, h)
        err_Heun[j] = max(err_Heun[j], abs(u_Heun-utrue(tnew)))

        # for first AB2 step, store fold and use RK2 for step; subsequently use AB2
        if (i == 0):
            fold_AB2 = f(t, u_AB2)
            u_AB2 = RK2_step(f, t, u_AB2, h)
        else:
            u_AB2, fold_AB2 = AB2_step(f, t, u_AB2, h, fold_AB2)
        err_AB2[j] = max(err_AB2[j], abs(u_AB2-utrue(tnew)))

        u_bE = bwd_Euler_step(f, f_u, t, u_bE,  h)
        err_bE[j] = max(err_bE[j], abs(u_bE-utrue(tnew)))

        u_trap = trap_step(f, f_u, t, u_trap,  h)
        err_trap[j] = max(err_trap[j], abs(u_trap-utrue(tnew)))


# output convergence results
print('Results for Fwd Euler:')
err = err_fE
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for RK2:')
err = err_RK2
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for Heun:')
err = err_Heun
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for AB2:')
err = err_AB2
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for Bwd Euler:')
err = err_bE
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for Trapezoidal:')
err = err_trap
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

# display true solution
plt.figure()
plt.plot(tvals, utrue(tvals))
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.show()

# end of script
