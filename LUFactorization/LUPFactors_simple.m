function [L,U,P] = LUPFactors_simple(A)
  % usage: [L,U,P] = LUPFactors_simple(A)
  %
  % Row-oriented LU factorization with simplistic row swapping; constructs the
  % factorization
  %       P A = LU  <=>  A = P^T L U
  % This function checks that A is square, and attempts to catch the case where A
  % is singular.
  %
  % Inputs:
  %   A - square n-by-n matrix
  %
  % Outputs:
  %   L - unit-lower-triangular matrix (n-by-n)
  %   U - upper-triangular matrix (n-by-n)
  %   P - permutation matrix (n-by-n)
  %
  % Daniel R. Reynolds
  % SMU Mathematics
  % Math 4315

  % check input
  [m,n] = size(A);
  if (m ~= n)
    error('LUPFactors_simple error: matrix must be square')
  end

  % set singularity tolerance
  tol = 1000*eps;

  % create output matrices
  U = A;
  L = eye(n);
  P = eye(n);
  for k=1:n-1                   % loop over pivots
    if (abs(U(k,k)) < tol)      % check for 'zero' pivot
      s=k;                      % find first 'nonzero' pivot
      for i = k+1:n
        if (abs(U(i,k)) > abs(U(s,k)))
          s = i;
          break
        end
      end
      if (abs(U(s,k)) < tol)      % check for singularity
        error('LUPFactors_simple error: matrix is [close to] singular')
      end
      for j=k:n                   % swap rows in U
        tmp = U(k,j);  U(k,j) = U(s,j);  U(s,j) = tmp;
      end
      for j=1:k-1                 % swap rows in L
        tmp = L(k,j);  L(k,j) = L(s,j);  L(s,j) = tmp;
      end
      for j=1:n                   % swap rows in P
        tmp = P(k,j);  P(k,j) = P(s,j);  P(s,j) = tmp;
      end
    end
    for i = k+1:n               % loop over remaining rows
      L(i,k) = U(i,k)/U(k,k);   % compute multiplier
      for j = k:n               % update remainder of matrix row
        U(i,j) = U(i,j) - L(i,k)*U(k,j);
      end
    end
  end
  return
