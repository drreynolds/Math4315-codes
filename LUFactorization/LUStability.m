% Script to test LUPFactors_simple, LUPFactors and LUPPFactors on a variety
% of ill-conditioned matrices.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% set matrix sizes for tests
nvals = [20, 40, 80, 160];

% loop over matrix sizes
for n = nvals

   fprintf('Testing stabilization approaches for linear system of dimension %i\n',n);

   % create the matrix, solution and right-hand side vectors
   A = vander(linspace(0.1,1,n)) + 0.0001*rand(n,n);
   n2 = floor(n/2);
   randrows = randi(n,n2,1);
   randcols = randi(n,n2,1);
   A(randrows,:) = diag(1000*rand(n2,1))*A(randrows,:);
   A(:,randcols) = A(:,randcols)*diag(1000*rand(n2,1));
   x = rand(n,1);
   b = A*x;

   % test LUPFactors_simple
   fprintf('  LUPFactors_simple:\n');
   [L,U,P] = LUPFactors_simple(A);
   fprintf('    norm(A - P^T L U) = %9.2e\n', norm(A - P'*L*U));
   x_comp = U\(L\(P*b));
   fprintf('    norm(x - x_comp) = %9.2e\n', norm(x - x_comp));

   % test LUPFactors
   fprintf('  LUPFactors:\n');
   [L,U,P] = LUPFactors(A);
   fprintf('    norm(A - P^T L U) = %9.2e\n', norm(A - P'*L*U));
   x_comp = U\(L\(P*b));
   fprintf('    norm(x - x_comp) = %9.2e\n', norm(x - x_comp));

   % test LUPPFactors
   fprintf('  LUPPFactors:\n');
   [L,U,P1,P2] = LUPPFactors(A);
   fprintf('    norm(A - P1^T L U P2^T) = %9.2e\n', norm(A - P1'*L*U*P2'));
   x_comp = P2*(U\(L\(P1*b)));
   fprintf('    norm(x - x_comp) = %9.2e\n', norm(x - x_comp));

end
