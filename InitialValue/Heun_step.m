function unew = Heun_step(f, told, uold, h)
% usage: unew = Heun_step(f, told, uold, h)
%
% Heun's Runge-Kutta of order 2 solver for one step of the
% ODE problem,
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
f1 = f(told, uold);

% set z1 and z2
z1 = uold;
z2 = uold + h*f1;

% get f(t+h,z2)
f2 = f(told+h, z2);

% update solution in time
unew = uold + h/2*(f1+f2);

% end of function
