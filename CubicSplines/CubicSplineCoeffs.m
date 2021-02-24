function z = CubicSplineCoeffs(t, y)
% Usage: z = CubicSplineCoeffs(t, y)
%
% This routine computes the coefficients of the natural interpolating cubic
% spline through the data values (t_k,y_k), k=0,...,n.
%
% Inputs:   t - array of interpolation knots
%           y - array of interpolation values
% Outputs:  z - cubic spline coefficients
%
% Daniel R. Reynolds
% SMU Mathematics
% Math4315

% check that dimensions of t and y match
if (size(t) ~= size(y))
  error('CubicSplineCoeffs error: node and data value array inputs must have identical size')
end

% get overall number of knots
n = length(t)-1;

% set knot spacing array
h = t(2:n+1) - t(1:n);

% set diagonal values
d = 2 * ( h(1:n-1) + h(2:n) );

% set right-hand side values
b = 6./h.*( y(2:n+1) - y(1:n) );
v = b(2:n) - b(1:n-1);

% set up tridiagonal linear system, A*z=V
A = zeros(n+1,n+1);
V = zeros(n+1,1);
for i=2:n
   A(i,i-1:i+1) = [h(i-1), d(i-1), h(i)];
   V(i) = v(i-1);
end

% set up first and last rows of linear system to enforce natural boundary conditions
A(1,1) = 1;
V(1) = 0;
A(n+1,n+1) = 1;
V(n+1) = 0;

% solve linear system for result (using tridiagonal solvers from homework 2)
[L,U] = LUFactorsTri(A);
z = BackwardSubTri(U, ForwardSubTri(L, V));

% end function
