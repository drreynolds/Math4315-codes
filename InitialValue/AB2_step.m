function [unew,fcur] = AB2_step(f, tcur, ucur, h, fold)
% usage: [unew,fcur] = AB2_step(f, tcur, ucur, h, fold)
%
% Adams-Bashforth method of order 2 for one step of the ODE
% problem,
%    u' = f(t,u), t in tspan,
%    u(t0) = u0.
%
% Inputs:  f = function for ODE right-hand side, f(t,u)
%          tcur = current time
%          ucur = current solution
%          h = time step size
%          fold = RHS evaluated at previous time step
% Outputs: unew = updated solution
%          fcur = RHS evaluated at current time step
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% get ODE RHS at this time step
fcur = f(tcur, ucur);

% update solution in time
unew = ucur + h/2*(3*fcur - fold);

% end of function
