% Script to test ForwardSub and BackwardSub on a variety of linear systems.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% set matrix sizes for tests
nvals = [50, 100, 200, 400];

% full-rank square matrix tests
for n = nvals

   fprintf('Testing with full-rank triangular matrices of dimension %i\n',n);

   % create the matrices
   L = tril(rand(n,n) + 2*eye(n));
   U = triu(rand(n,n) + 2*eye(n));

   % create solution vector
   x = rand(n,1);

   % solve triangular linear systems
   x_bs = BackwardSub(U, U*x);
   x_fs = ForwardSub(L, L*x);

   % output results
   fprintf('   BackwardSub error = %g\n', norm(x-x_bs));
   fprintf('   ForwardSub error  = %g\n', norm(x-x_fs));

end

% ensure that rank-deficient case fails
n = 100;
fprintf('Testing with rank-deficient triangular matrices (should fail)\n');
L = tril(rand(n,n));  L(n-3,n-3) = eps;
U = triu(rand(n,n));  U(n-5,n-5) = eps;
x = rand(n,1);
try
  x_bs = BackwardSub(U, U*x);
catch
  fprintf('   rank deficiency correctly caught by BackwardSub\n');
end
try
  x_fs = ForwardSub(L, L*x);
catch
  fprintf('   rank deficiency correctly caught by ForwardSub\n');
end
