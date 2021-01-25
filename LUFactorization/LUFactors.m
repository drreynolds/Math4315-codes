function [L,U] = LUFactors(A)
  % usage: [L,U] = LUFactors(A)
  %
  % Row-oriented naive LU factorization; constructs the factorization
  %       A = LU
  % This function checks that A is square, and attempts to catch the case where A
  % is singular or would require partial pivoting.
  %
  % Inputs:
  %   A - square n-by-n matrix
  %
  % Outputs:
  %   L - unit-lower-triangular matrix (n-by-n)
  %   U - upper-triangular matrix (n-by-n)
  %
  % Daniel R. Reynolds
  % SMU Mathematics
  % Math 4315

  % check input
  [m,n] = size(A);
  if (m ~= n)
    error('LUFactors error: matrix must be square')
  end

  % set singularity tolerance
  tol = 1000*eps;

  % create output matrices
  U = A;
  L = eye(n);
  for k=1:n-1                   % loop over pivots
    if (abs(U(k,k)) < tol)      % check for failure
      error('LUFactors error: factorization failure')
    end
    for i = k+1:n               % loop over remaining rows
      L(i,k) = U(i,k)/U(k,k);   % compute multiplier
      for j = k:n               % update remainder of matrix row
        U(i,j) = U(i,j) - L(i,k)*U(k,j);
      end
    end
  end
  return
