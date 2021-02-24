function [y] = ForwardSubTri(L,b)
  % usage: [y] = ForwardSubTri(L,b)
  %
  % Row-oriented forward substitution to solve the lower-triangular, 'tridiagonal'
  % linear system
  %       L y = b
  % This function does not ensure that L has the correct nonzero structure.  It does,
  % however, attempt to catch the case where L is singular.
  %
  % Inputs:
  %   L - square n-by-n matrix (assumed lower triangular and 'tridiagonal')
  %   b - right-hand side vector (n-by-1)
  %
  % Outputs:
  %   y - solution vector (n-by-1)
  %
  % Daniel R. Reynolds
  % SMU Mathematics
  % Math 4315

  % check inputs
  [m,n] = size(L);
  if (m ~= n)
    error('ForwardSubTri error: matrix must be square')
  end
  [p,q] = size(b);
  if ((p ~= n) || (q ~= 1))
    error('ForwardSubTri error: right-hand side vector has incorrect dimensions')
  end
  if (min(abs(diag(L))) < 100*eps)
    error('ForwardSubTri error: matrix is [close to] singular')
  end

  % create output vector
  y = b;

  % perform tridiagonal forward-subsitution algorithm
  for i=1:n
    if (i>1)
      y(i) = y(i) - L(i,i-1)*y(i-1);
    end
    y(i) = y(i)/L(i,i);
  end

  return
