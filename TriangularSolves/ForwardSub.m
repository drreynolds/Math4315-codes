function [y] = ForwardSub(L,b)
  % usage: [y] = ForwardSub(L,b)
  %
  % Row-oriented forward substitution to solve the lower-triangular
  % linear system
  %       L y = b
  % This function does not ensure that L is lower-triangular.  It does,
  % however, attempt to catch the case where L is singular.
  %
  % Inputs:
  %   L - square n-by-n matrix (assumed lower triangular)
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
    error('ForwardSub error: matrix must be square')
  end
  [p,q] = size(b);
  if ((p ~= n) || (q ~= 1))
    error('ForwardSub error: right-hand side vector has incorrect dimensions')
  end
  if (min(abs(diag(L))) < 100*eps)
    error('ForwardSub error: matrix is [close to] singular')
  end

  % create output vector
  y = b;

  % perform forward-subsitution algorithm
  for i=1:n
    for j=1:i-1
      y(i) = y(i) - L(i,j)*y(j);
    end
    y(i) = y(i)/L(i,i);
  end

  return
