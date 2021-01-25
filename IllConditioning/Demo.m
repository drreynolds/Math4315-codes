% Script to demonstrate effects of matrix conditioning in floating-point arithmetic.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% set matrix sizes for tests
nvals = [6 8 10 12 14];

% run tests for each matrix size
for n = nvals

  % create matrix, solution and right-hand side vector
  A = hilb(n);
  x = rand(n,1);
  b = A*x;

  % ouptut condition number
  fprintf('Hilbert matrix of dimension %i: condition number = %g\n', n, cond(A));

  % solve the linear system
  S = warning('off','MATLAB:nearlySingularMatrix');
  x_comp = A\b;

  % output relative solution error
  fprintf('  relative solution error = %g\n', norm(x-x_comp)/norm(x))

end
