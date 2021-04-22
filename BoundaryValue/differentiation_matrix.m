function [x,D] = differentiation_matrix(n, a, b, order, deriv)
% Usage: [x,D] = differentiation_matrix(n, a, b, order, deriv)
%
% Utility to compute differentiation matrix of specified order
% of accuracy and derivative order, over an interval [a,b].
%
% Inputs:  n = number of intervals to use
%          a,b = interval to discretize
%          order = differentiation matrix type:
%                   0 = Chebyshev [spectral convergence]
%                   1 = O(h) finite-difference over regular mesh
%                   2 = O(h^2) finite-difference over regular mesh
%          deriv = derivative order {1,2}
% Outputs: x = column vector containing partition of [a,b]
%          D = differentiation matrix
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% check for a sufficient number of intervals
if (n < 2)
   error('insufficient number of intervals')
end

% ensure that order and deriv are integers
order = int8(order);
deriv = int8(deriv);

% check for valid order
if ((order < 0) || (order > 2))
   error('invalid order selected')
end

% construct matrix based on desired differentiation order

if (order == 0)   % Chebyshev [based off of 'diffcheb' function 10.2.2 from the book]

  % set Chebyshev nodes in [-1,1]
  x = -cos( (0:n)'*pi/n );

  % create base differentiation matrix
  Dbase = zeros(n+1);
  c = [2; ones(n-1,1); 2];
  i = (0:n)';
  for j=0:n
    num = c(i+1).*(-1).^(i+j);
    den = c(j+1)*(x - x(j+1));
    Dbase(:,j+1) = num./den;
    Dbase(j+1,j+1) = 0;
  end
  Dbase = Dbase - diag(sum(Dbase,2));

  % remap to interval [a,b]
  x = a + (b-a)/2*(x+1);
  Dbase = (2/(b-a))*Dbase;

  % construct output matrix through multiplication
  D = Dbase;
  for i=2:deriv
    D = D*Dbase;
  end

else             % finite-difference

  % set uniform nodes and corresponding h
  x = linspace(a,b,n+1)';
  h = (b-a)/n;

  % first order, first derivative
  if ((order == 1) && (deriv == 1))

    D = diag(ones(n,1),1) - diag(ones(n+1,1), 0);
    D(n+1,n:n+1) = [-1, 1];
    D = (1/h)*D;

  % second order, first derivative
  elseif ((order == 2) && (deriv == 1))

    D = 1/2*(diag(ones(n,1),1) - diag(ones(n,1),-1));
    D(1,1:3) = [-3/2, 2, -1/2];
    D(n+1,n-1:n+1) = [1/2, -2, 3/2];
    D = (1/h)*D;

  % second order, second derivative
  elseif ((order == 2) && (deriv == 2))

    D = diag(ones(n,1),1) + diag(ones(n,1),-1) - 2*diag(ones(n+1,1),0);
    D(1,1:4) = [2, -5, 4, -1];
    D(n+1,n-2:n+1) = [-1, 4, -5, 2];
    D = (1/h/h)*D;

 % all other choices are not implemented
  else
    error('invalid order/deriv selection for finite-difference matrix')
  end

end


% end of function
