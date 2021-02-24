function p = LinearSpline(x,t,y)
% usage: p = LinearSpline(x,t,y)
%
% Function to evaluate the linear spline defined by the data values
% (t_k,y_k), k=0,...,n, at the point(s) x.
%
% Inputs:   x - point(s) to evaluate linear spline
%           t - array of interpolation nodes
%           y - array of interpolation values
% Outputs:  p - value of linear spline at point(s) x
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% check that dimensions of t and y match
if (size(t) ~= size(y))
  error('LinearSpline error: node and data value array inputs must have identical size')
end

% get overall number of nodes
n = length(t)-1;

% initialize output
p = zeros(size(x));

% evaluate p by adding in contributions from each hat function

%   left-most hat function
p = p + y(1)*max(0,(t(2)-x)/(t(2)-t(1)));

%   right-most hat function
if (n > 0)
  p = p + y(n+1)*max(0,(x-t(n))/(t(n+1)-t(n)));
end

% intermediate hat functions
for k=2:n
  p = p + y(k)*max(0,min((x-t(k-1))/(t(k)-t(k-1)),(t(k+1)-x)/(t(k+1)-t(k))));
end

% end function
