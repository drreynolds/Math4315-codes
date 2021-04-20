function [coeffs,errorterm] = fd_maker(stencil, deriv)
% Usage: [coeffs,errorterm] = fd_maker(stencil, deriv)
%
% Utility to compute classical finite difference approximation to a requested derivative
% using a specified set of nodes.  We assume that all nodes are evenly spaced, with a
% spacing of 'h', and that the derivative is requested at the node "0", meaning that the
% location of each node may be uniquely specified by a "stencil" of offsets from the
% derivative location.
%
% Inputs:  stencil = array of integer offsets from node "0" that
%                     will be used in approximation, e.g. [-1, 0, 1]
%                     for f(x-h), f(x) and f(x+h)
%          deriv = integer specifying the desired derivative
%
% Outputs: coeffs = row vector of finite-difference coefficients s.t.
%                      f^(deriv) \approx \sum coeffs(i)*f(x+stencil(i)*h)
%          errorterm = leading error term in derivative approximation
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% check for a sufficient number of coefficients
n = length(stencil);
if (deriv > n-1)
   error('not enough stencil entries for requested derivative')
end

% convert 'stencil' to a row vector
[rows,cols] = size(stencil);
if (rows > cols)
   stencil = stencil';
end

% create and solve symbolic linear system for coefficients
syms h;
A = zeros(n,n);
for i=1:n
   A(i,:) = stencil.^(i-1);
end
fact = h^(-deriv);
for i=1:deriv
   fact = fact*i;
end
b(deriv+1,1) = fact;
if (deriv+1 < n)
   b(n,1) = 0;
end
c = A\b;

% transpose coefficients for return
for i=1:n
   coeffs(1,i) = c(i);
end



% check conditions (up to twice as far along) to find error term
%   create larger linear system of conditions
A = zeros(2*n,n);
for i=1:2*n
   A(i,:) = stencil.^(i-1);
end
b(deriv+1,1) = fact;
b(2*n,1) = 0;

%   determine which equations fail, scale by corresponding factor of h
err = A*c-b;
for i=1:2*n
   err(i) = err(i)*h^(i-1);
end

%   identify the leading nonzero error term and store for output
h = 10;
errtest = eval(err);
for i=1:2*n
   if (abs(errtest(i)) > 10*eps)
      errorterm = err(i);
      break;
   end
end

% end of function
