% Script to compare polynomial interpolants using various choices of node types.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% remove all existing variables
clear

% set numbers of nodes for tests
nvals = [5, 10, 20, 40];

% set function and interval that we will interpolate
f  = @(x) 1./(1+x.^2);
a = -5;
b = 5;

% set evaluation points to compare interpolants
x = linspace(a, b, 2001);

% initialize plots
figure(1)    % uniform nodes
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(2)
hold on
figure(3)    % random nodes
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(4)
hold on
figure(5)    % Chebyshev nodes of first kind
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(6)
hold on
figure(7)    % Chebyshev nodes of second kind
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(8)
hold on

% loop over node numbers
for n = nvals

   fprintf('Testing with n = %i\n', n);

   % uniformly-spaced nodes
   figure(1)
   t = linspace(a,b,n+1);
   p = Lagrange(t, f(t), x);
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(2)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

   % random nodes
   figure(3)
   t = a + (b-a)*rand(1,n+1);
   p = Lagrange(t, f(t), x);
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(4)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

   % Chebyshev nodes of the first kind
   figure(5)
   t = (a+b)/2 + (b-a)/2*cos((2*(0:n)+1)/(2*n+2)*pi);
   p = Lagrange(t, f(t), x);
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(6)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

   % Chebyshev nodes of the second kind
   figure(7)
   t = (a+b)/2 - (b-a)/2*cos((0:n)*pi/n);
   p = Lagrange(t, f(t), x);
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(8)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

end

% finalize plots
figure(1)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Uniformly-spaced nodes')

figure(2)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Uniformly-spaced node error')

figure(3)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Random nodes')

figure(4)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Random node error')

figure(5)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Chebyshev nodes of first kind')

figure(6)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Chebyshev node of first kind error')

figure(7)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Chebyshev nodes of second kind')

figure(8)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Chebyshev node of second kind error')

% end of script
