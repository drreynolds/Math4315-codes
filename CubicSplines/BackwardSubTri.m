function [x] = BackwardSubTri(U,y)
  % usage: [x] = BackwardSubTri(U,y)
  %
  % Row-oriented backward substitution to solve the upper-triangular, 'tridiagonal'
  % linear system
  %       U x = y
  % This function does not ensure that U has the correct nonzero structure.  It does,
  % however, attempt to catch the case where U is singular.
  %
  % Inputs:
  %   U - square n-by-n matrix (assumed upper triangular and 'tridiagonal')
  %   y - right-hand side vector (n-by-1)
  %
  % Outputs:
  %   x - solution vector (n-by-1)
  %
  % Daniel R. Reynolds
  % SMU Mathematics
  % Math 4315

  % check inputs
  [m,n] = size(U);
  if (m ~= n)
    error('BackwardSubTri error: matrix must be square')
  end
  [p,q] = size(y);
  if ((p ~= n) || (q ~= 1))
    error('BackwardSubTri error: right-hand side vector has incorrect dimensions')
  end
  if (min(abs(diag(U))) < 100*eps)
    error('BackwardSubTri error: matrix is [close to] singular')
  end

  % create output vector
  x = y;

  % perform forward-subsitution algorithm
  for i=n:-1:1
    if (i<n)
      x(i) = x(i) - U(i,i+1)*x(i+1);
    end
    x(i) = x(i)/U(i,i);
  end

  return
