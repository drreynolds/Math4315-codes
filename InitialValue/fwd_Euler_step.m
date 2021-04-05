function unew = fwd_Euler_step(f, told, uold, h)
% usage: unew = fwd_Euler_step(f, told, uold, h)
%
% Forward Euler solver for one step of the ODE problem,
%    u' = f(t,u), t in tspan,
%    u(t0) = u0.
%
% Inputs:  f = function for ODE right-hand side, f(t,u)
%          told = current time
%          uold = current solution
%          h = time step size
% Outputs: unew = updated solution
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% call f to get ODE RHS at this time step
fn = f(told, uold);

% update solution in time
unew = uold + h*fn;

% end of function
