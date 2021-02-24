% Script to test LinearSpline on a variety of data sets
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% set numbers of nodes for tests
nvals = [5, 10, 20, 40];

% set function and interval that we will interpolate
f = @(x) x+sin(2*x);
a = 0;
b = 4;

% create evaluation points for plots
x = linspace(a,b,201);

% initialize plots
figure(1)
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(2)
hold on

% loop over node numbers
for n = nvals

   fprintf('Testing with n = %i\n',n);

   % create the nodes and data
   t = linspace(a,b,n+1);
   y = f(t);

   % fill p by calling LinearSpline over the evaluation points
   p = LinearSpline(x,t,y);

   % compute maximum error in interpolation
   err = norm(f(x)-p, inf);

   % add interpolant to plot
   figure(1)
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,err))

   % add error to plot
   figure(2)
   plot(x, abs(f(x)-p), 'DisplayName', sprintf('|f(x) - p_{%i}(x)|',n))

end

% finalize plots
figure(1)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Linear Spline Interpolants')

figure(2)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Linear Spline Error')
