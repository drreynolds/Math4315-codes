function T = Chebyshev(x,k)
% Usage: T = Chebyshev(x,k)
%
% Function to evaluate the kth Chebyshev polynomial at a point x.
%
% Inputs:   x - evaluation point(s)
%           k - Chebyshev polynomial index
% Outputs:  T - value(s) of T_k(x)
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% evaluate using cosine formulation
T = cos(k * acos(x));

% end of function
