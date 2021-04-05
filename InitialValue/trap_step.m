function unew = trap_step(f, f_u, told, uold, h)
% Usage: unew = trap_step(f, f_u, told, uold, h)
%
% Trapezoidal (2nd-order Adams-Moulton) solver for one step of the ODE problem,
%    u' = f(t,u), t in tspan,
%    u(t0) = u0.
%
% Inputs:  f = function for ODE right-hand side, f(t,u)
%          f_u = function for ODE Jacobian, f_u(t,u)
%          told = current time
%          uold = current solution
%          h = time step size
% Outputs: unew = updated solution
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% set nonlinear solver parameters
maxit = 20;
rtol = 1e-9;
atol = 1e-12;
output = false;

% create implicit residual and Jacobian functions
fold = f(told, uold);
I = eye(length(uold));
F = @(unew) unew - uold - 0.5*h*(f(told+h, unew) + fold);
A = @(unew) I - 0.5*h*f_u(told+h, unew);

% perform implicit solve
[unew, its] = newton(F, A, uold, maxit, rtol, atol, output);

% end of function
