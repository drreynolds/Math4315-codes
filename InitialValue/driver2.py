#!/usr/bin/env python3
#
# Script to test various ODE solvers for the stiff IVP system
#    [u1,u2]' = [[-100, 1], [-1, -100]]*[u1,u2] + [100*sin(10*t), 100*cos(10*t)],   0<t<1
#    [u1,u2](0) = [10, 20].
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from ivp_methods import *

# ODE RHS and Jacobian, initial condition, etc.
A = numpy.array([[-100.0, 1.0], [-1.0, -100.0]])
def f(t,u):
    return A@u + [100*numpy.sin(10*t), 100*numpy.cos(10*t)]
def f_u(t,u):
    return A
u0  = numpy.array([10.0, 20.0])
t0  = 0.0
Tf  = 1.0
hvals = numpy.array([0.5**5, 0.5**6, 0.5**7, 0.5**8, 0.5**9, 0.5**10, 0.5**11])

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

    # construct reference solution
    ivp_ref = integrate.solve_ivp(f, (t0,Tf), u0, t_eval=tvals, jac=f_u,
                                rtol=1e-6, atol=1e-10, method='Radau')
    u_ref = ivp_ref.y

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
        err_fE[j] = max(err_fE[j], numpy.linalg.norm(u_fE-u_ref[:,i+1]))

        u_RK2 = RK2_step(f, t, u_RK2,  h)
        err_RK2[j] = max(err_RK2[j], numpy.linalg.norm(u_RK2-u_ref[:,i+1]))

        u_Heun = Heun_step(f, t, u_Heun, h)
        err_Heun[j] = max(err_Heun[j], numpy.linalg.norm(u_Heun-u_ref[:,i+1]))

        # for first AB2 step, store fold and use RK2 for step; subsequently use AB2
        if (i == 0):
            fold_AB2 = f(t, u_AB2)
            u_AB2 = RK2_step(f, t, u_AB2, h)
        else:
            u_AB2, fold_AB2 = AB2_step(f, t, u_AB2, h, fold_AB2)
        err_AB2[j] = max(err_AB2[j], numpy.linalg.norm(u_AB2-u_ref[:,i+1]))

        u_bE = bwd_Euler_step(f, f_u, t, u_bE,  h)
        err_bE[j] = max(err_bE[j], numpy.linalg.norm(u_bE-u_ref[:,i+1]))

        u_trap = trap_step(f, f_u, t, u_trap,  h)
        err_trap[j] = max(err_trap[j], numpy.linalg.norm(u_trap-u_ref[:,i+1]))



# output convergence results
print('Results for Fwd Euler:')
err = err_fE
print("   h = %12g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %12g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for RK2:')
err = err_RK2
print("   h = %12g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %12g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for Heun:')
err = err_Heun
print("   h = %12g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %12g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for AB2:')
err = err_AB2
print("   h = %10g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %10g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for Bwd Euler:')
err = err_bE
print("   h = %12g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %12g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

print('Results for Trapezoidal:')
err = err_trap
print("   h = %12g,  err = %.2e" % (hvals[0], err[0]))
for i in range(1,len(hvals)):
    print("   h = %12g,  err = %.2e,  rate = %g" %
          (hvals[i], err[i], numpy.log(err[i]/err[i-1])/numpy.log(hvals[i]/hvals[i-1])) )

# display reference solution
plt.figure()
plt.plot(ivp_ref.t, ivp_ref.y[0,:], ivp_ref.t, ivp_ref.y[1,:])
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.legend(('$u_1(t)$','$u_2(t)$'))
plt.show()

# end of script
