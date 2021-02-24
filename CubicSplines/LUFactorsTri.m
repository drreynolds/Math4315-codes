function [L,U] = LUFactorsTri(T)
  % usage: [L,U] = LUFactorsTri(T)
  %
  % Row-oriented naive LU factorization for tridiagonal matrices; constructs the 
  % factorization
  %       T = LU
  % This function checks that T is square, and attempts to catch the case where T
  % is singular.
  %
  % Inputs:
  %   T - tridiagonal n-by-n matrix
  %
  % Outputs:
  %   L - unit-lower-triangular 'tridiagonal' matrix (n-by-n)
  %   U - upper-triangular 'tridiagonal' matrix (n-by-n)
  %
  % Daniel R. Reynolds
  % SMU Mathematics
  % Math 4315

  % check input
  [m,n] = size(T);
  if (m ~= n)
    error('LUFactorsTri error: matrix must be square')
  end

  % set singularity tolerance
  tol = 1000*eps;

  % create output matrices
  U = T;
  L = eye(n);
  for k=1:n-1                     % loop over pivots
    if (abs(U(k,k)) < tol)        % check for failure
      error('LUFactors error: factorization failure')
    end
    L(k+1,k) = U(k+1,k)/U(k,k);   % compute multiplier for next row
    for j = k:k+1                 % update remainder of matrix row
      U(k+1,j) = U(k+1,j) - L(k+1,k)*U(k,j);
    end
  end
  return
