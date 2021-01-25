% Script to test LUFactors and LUPFactors on a variety of linear systems.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% set matrix sizes for tests
nvals = [50, 100, 200, 400];

% set tolerance for tests
tol = sqrt(eps);

% tests for diagonally-dominant matrices
for n = nvals

   fprintf('Testing LUFactors with diagonally-dominant matrix of dimension %i\n',n);

   % create the matrix
   A = rand(n,n) + n*eye(n);

   % construct and test LU factorization
   [L,U] = LUFactors(A);
   checks_out = true;
   if ((norm(L - tril(L))>tol) || (norm(diag(L)-ones(n,1)) > tol))
     checks_out = false;
     fprintf("  LUFactors failure: L is not unit lower-triangular\n");
   end
   if (norm(U - triu(U))>tol)
     checks_out = false;
     fprintf("  LUFactors failure: U is not upper-triangular\n");
   end
   if (norm(A-L*U)>tol)
     checks_out = false;
     fprintf("  LUFactors failure: A ~= LU\n");
   end
   if (checks_out)
     fprintf("  LUFactors passes all tests\n")
   end

end


% tests for nonsingular (but non-diagonally-dominant) matrices
for n = nvals

   fprintf('Testing LUPFactors with non-diagonally-dominant matrix of dimension %i\n',n);

   % create the matrix
   A = rand(n,n);
   for i=1:n
     A(i,n-i+1) = A(i,n-i+1) + n;
   end

   % construct and test LUP factorization
   [L,U,P] = LUPFactors(A);
   checks_out = true;
   if ((norm(L - tril(L))>tol) || (norm(diag(L)-ones(n,1)) > tol))
     checks_out = false;
     fprintf("  LUPFactors failure: L is not unit lower-triangular\n");
   end
   if (norm(U - triu(U))>tol)
     checks_out = false;
     fprintf("  LUPFactors failure: U is not upper-triangular\n");
   end
   if (norm(P'*P - eye(n))>tol)
     checks_out = false;
     fprintf("  LUPFactors failure: P is not a permutation matrix\n");
   end
   if (norm(A-P'*L*U)>tol)
     checks_out = false;
     fprintf("  LUPFactors failure: A ~= P^T L U\n");
   end
   if (checks_out)
     fprintf("  LUPFactors passes all tests\n")
   end

end


% ensure that singular case fails
n = 100;
fprintf('Testing with singular matrices (should fail)\n');
A = rand(n,n);
A(:,n-20) = A(:,1) - 4*A(:,10);
try
  [L,U] = LUFactors(A);
catch
  fprintf('   singularity correctly caught by LUFactors\n');
end
try
   [L,U,P] = LUPFactors(A);
catch
  fprintf('   singularity correctly caught by LUPFactors\n');
end
