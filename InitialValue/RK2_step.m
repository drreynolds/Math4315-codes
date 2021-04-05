function unew = RK2_step(f, told, uold, h)
% usage: unew = RK2_step(f, told, uold, h)
%
% Runge-Kutta of order 2 solver for one step of the ODE
% problem,
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

% get ODE RHS at this time step
f1 = f(told, uold);

% set z1 and z2
z1 = uold;
z2 = uold + h/2*f1;

% get f(t+h/2,z2)
f2 = f(told+h/2, z2);

% update solution in time
unew = uold + h*f2;

% end of function
