function p = Legendre(x,k)
% Usage: p = Legendre(x,k)
%
% Function to evaluate the kth Legendre polynomial at a point x.
%
% Inputs:   x - evaluation point(s)
%           k - Legendre polynomial index
% Outputs:  p - value(s) of p_k(x)
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% quick return for first two Legendre polynomials
if (k == 0)
   p = ones(size(x));
   return;
end
if (k == 1)
   p = x;
   return;
end

% initialize 3-term recurrence
p0 = ones(size(x));
p1 = x;

% perform recurrence to evaluate p, and update 'old' values
for i=2:k
  p = (2*i-1)/i*x.*p1 - (i-1)/i*p0;
  p0 = p1;
  p1 = p;
end

% end of function
