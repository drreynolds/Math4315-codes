function [L,U,P1,P2] = LUPPFactors(A)
  % usage: [L,U,P1,P2] = LUPPFactors(A)
  %
  % Row-oriented LU factorization with complete pivoting; constructs the factorization
  %       P1 A P2 = LU  <=>  A = P1^T L U P2^T
  % This function checks that A is square, and attempts to catch the case where A
  % is singular.
  %
  % Inputs:
  %   A - square n-by-n matrix
  %
  % Outputs:
  %   L  - unit-lower-triangular matrix (n-by-n)
  %   U  - upper-triangular matrix (n-by-n)
  %   P1 - permutation matrix (n-by-n)
  %   P2 - permutation matrix (n-by-n)
  %
  % Daniel R. Reynolds
  % SMU Mathematics
  % Math 4315

  % check input
  [m,n] = size(A);
  if (m ~= n)
    error('LUPPFactors error: matrix must be square')
  end

  % set singularity tolerance
  tol = 1000*eps;

  % create output matrices
  U = A;
  L = eye(n);
  P1 = eye(n);
  P2 = eye(n);
  for k=1:n-1                   % loop over pivots
    si=k;                       % determine pivot position
    sj=k;
    for i = k:n
      for j = k:n
        if (abs(U(i,j)) > abs(U(si,sj)))
          si = i;
          sj = j;
        end
      end
    end
    if (abs(U(si,sj)) < tol)    % check for singularity
      error('LUPPFactors error: matrix is [close to] singular')
    end
    for j=k:n                   % swap rows in U
      tmp = U(k,j);  U(k,j) = U(si,j);  U(si,j) = tmp;
    end
    for j=1:k-1                 % swap rows in L
      tmp = L(k,j);  L(k,j) = L(si,j);  L(si,j) = tmp;
    end
    for j=1:n                   % swap rows in P1
      tmp = P1(k,j);  P1(k,j) = P1(si,j);  P1(si,j) = tmp;
    end
    for i=1:n                   % swap columns in U
      tmp = U(i,k);  U(i,k) = U(i,sj);  U(i,sj) = tmp;
    end
    for i=1:n                   % swap columns in P2
      tmp = P2(i,k);  P2(i,k) = P2(i,sj);  P2(i,sj) = tmp;
    end
    for i = k+1:n               % loop over remaining rows
      L(i,k) = U(i,k)/U(k,k);   % compute multiplier
      for j = k:n               % update remainder of matrix row
        U(i,j) = U(i,j) - L(i,k)*U(k,j);
      end
    end
  end
  return
