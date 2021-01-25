function [x] = BackwardSub(U,y)
  % usage: [x] = BackwardSub(U,y)
  %
  % Row-oriented backward substitution to solve the upper-triangular
  % linear system
  %       U x = y
  % This function does not ensure that U is upper-triangular.  It does,
  % however, attempt to catch the case where U is singular.
  %
  % Inputs:
  %   U - square n-by-n matrix (assumed upper triangular)
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
    error('BackwardSub error: matrix must be square')
  end
  [p,q] = size(y);
  if ((p ~= n) || (q ~= 1))
    error('BackwardSub error: right-hand side vector has incorrect dimensions')
  end
  if (min(abs(diag(U))) < 100*eps)
    error('BackwardSub error: matrix is [close to] singular')
  end

  % create output vector
  x = y;

  % perform forward-subsitution algorithm
  for i=n:-1:1
    for j=i+1:n
      x(i) = x(i) - U(i,j)*x(j);
    end
    x(i) = x(i)/U(i,i);
  end

  return
