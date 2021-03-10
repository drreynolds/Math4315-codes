function p = Lagrange(t,y,x)
% Usage: p = Lagrange(t,y,x)
%
% Function to evaluate the Lagrange interpolating polynomial p(x)
% defined in the Lagrange basis by
%     p(x) = y(1)*l1(x) + y(2)*l2(x) + ... + y(n)*ln(x).
%
% Inputs:   t - array of interpolation nodes
%           y - array of interpolation values
%           x - point(s) to evaluate Lagrange interpolant
% Outputs:  p - value of Lagrange interpolant at point(s) x
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% check inputs
if (length(t) ~= length(y))
   error('Lagrange error: (t,y) have different sizes');
end
n = length(t);

% initialize output
p = zeros(size(x));

% iterate over Lagrange basis functions
for k=1:n

   % initialize l (the kth Lagrange basis function)
   l = ones(size(x));

   % iterate over data to construct l(x)
   for j=1:n
      % exclude the k-th data point
      if (j ~= k)
          l = l.*(x-t(j))/(t(k)-t(j));
      end
   end

   % add contribution from this basis function (and data value) into p
   p = p + y(k)*l;

end

% end of function
