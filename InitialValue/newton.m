function [x, its] = newton(Ffun, Jfun, x, maxit, rtol, atol, output)
% Usage: [x, its] = newton(Ffun, Jfun, x, maxit, rtol, atol, output)
%
% This routine uses the Newton method to approximate a root of
% the nonlinear system of equations F(x)=0.  The iteration ceases
% when the following condition is met:
%
%    ||xnew - xold|| < atol + rtol*||xnew||
%
% inputs:   Ffun     nonlinear residual function
%           Jfun     Jacobian function
%           x        initial guess at solution
%           maxit    maximum allowed number of iterations
%           rtol     relative solution tolerance
%           atol     absolute solution tolerance
%           output   flag (true/false) to output iteration history
% outputs:  x        approximate solution
%           its      number of iterations taken
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% check input arguments
if (floor(maxit) < 1)
  fprintf('newton: maxit = %i < 1. Resetting to 10\n',floor(maxit));
  maxit = 10;
end
if (rtol < 10*eps)
  fprintf('newton: rtol = %g < %g. Resetting to %g\n', rtol, 10*eps, 10*eps)
  rtol = 10*eps;
end
if (atol < 0)
  fprintf('newton: atol = %g < 0. Resetting to %g\n', atol, 1e-15)
  atol = 1e-15;
end

% evaluate initial residual
F = Ffun(x);

% begin iteration
for its=1:maxit

   % evaluate derivative
   J = Jfun(x);

   % compute Newton update, new guess at solution, new residual
   h = J\F;
   x = x - h;
   F = Ffun(x);

   % check for convergence and output diagnostics
   hnorm = norm(h);
   xnorm = norm(x);
   if (output)
      fprintf('   iter %3i, \t||h|| = %g, \ttol = %g\n', its, hnorm, atol + rtol*xnorm);
   end
   if (hnorm < atol + rtol*xnorm)
      break;
   end

end

% end of function
