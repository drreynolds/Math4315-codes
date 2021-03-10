function v = L2InnerProduct(f,g,w,a,b)
% Usage: v = L2InnerProduct(f,g,w,a,b)
%
% Function to evaluate the weighted L^2 inner product between two functions, f and g, over an interval [a,b], based on the weight function w(x).
%
% Inputs:   f - function handle
%           g - function handle
%           w - function handle (assumed to have strictly positive values)
%           a - left endpoint of interval
%           b - right endpoint of interval
% Outputs:  v - value of inner product
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% ensure that interval is valid
if (b <= a)
   error('L2InnerProduct error: invalid interval');
end

% set integrand (allow array-valued argument)
integrand = @(x) f(x).*g(x).*w(x);

% approximate integral over [a,b]
v = integral(integrand, a, b);

% end of function
