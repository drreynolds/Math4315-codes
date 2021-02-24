function s = CubicSplineEvaluate(t, y, z, x)
% Usage: s = CubicSplineEvaluate(t, y, z, x)
%
% This routine evaluates the cubic spline defined by the knots, t, the data
% values, y, and the coefficients, z, at the point x.
%
% Inputs:   t - array of interpolation knots
%           y - array of interpolation values
%           z - cubic spline coefficients
%           x - evaluation point(s)
% Output:   s - value of cubic spline at point(s) x
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% check input arguments
if ((length(t) ~= length(y)) || (length(t) ~= length(z)))
   error('CubicSplineEvaluate error: (t,y,z) have different sizes');
end

% get overall number of knots
n = length(t)-1;

% create output
s = zeros(size(x));

% evaluate spline for each entry in x
for j=1:length(x)

  % determine spline interval for this x value
  if (x(j) < t(1))
    i = 1;
  elseif (x(j) > t(n+1))
    i = n;
  else
    for i=1:n
      if ( (x(j) >= t(i)) && (x(j) < t(i+1)) )
        break
      end
    end
  end

  % set subinterval width
  h = t(i+1) - t(i);

  % evaluate spline
  s(j) = z(i)/(6*h)*(t(i+1)-x(j))^3 ...
         + z(i+1)/(6*h)*(x(j)-t(i))^3  ...
         + (y(i+1)/h - z(i+1)*h/6)*(x(j)-t(i)) ...
         + (y(i)/h - z(i)*h/6)*(t(i+1)-x(j));
end

% end function
