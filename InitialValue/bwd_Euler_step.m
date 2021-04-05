function unew = bwd_Euler_step(f, f_u, told, uold, h)
% Usage: unew = bwd_Euler_step(f, f_u, told, uold, h)
%
% Backward Euler solver for one step of the ODE problem,
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
I = eye(length(uold));
F = @(unew) unew - uold - h*f(told+h, unew);
A = @(unew) I - h*f_u(told+h, unew);

% perform implicit solve
[unew, its] = newton(F, A, uold, maxit, rtol, atol, output);

% end of function
