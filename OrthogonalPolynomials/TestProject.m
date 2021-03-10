% Script to compare orthogonal polynomial projections.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% remove all existing variables
clear

% set interval and weight functions
a = -1;
b = 1;
wL = @(x) ones(size(x));
wC = @(x) 1./sqrt(1-x.^2);

% set polynomial degrees for tests
nvals = [5, 10, 15, 20];

% set function that we will interpolate
f  = @(x) 1./(1+x.^2);

% set evaluation points for plots
x = linspace(a, b, 2001);

% initialize plots
figure(1)    % Chebyshev interpolant
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(2)
hold on
figure(3)    % Legendre projection
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(4)
hold on
figure(5)    % Chebyshev projection
plot(x, f(x), 'DisplayName', 'f(x)')
hold on
figure(6)
hold on

% loop over polynomial degrees
for n = nvals

   fprintf('Testing with n = %i\n', n);

   % Chebyshev interpolant
   figure(1)
   t = cos((2*(0:n)+1)/(2*n+2)*pi);
   p = Lagrange(t, f(t), x);
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(2)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

   % Legendre projection
   figure(3)
   p = zeros(size(x));
   for k=0:n
     pk = @(x) Legendre(x,k);
     c = L2InnerProduct(pk,f,wL,a,b) * (k+1/2);
     p = p + c*pk(x);
   end
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(4)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

   % Chebyshev projection
   figure(5)
   p = zeros(size(x));
   for k=0:n
     pk = @(x) Chebyshev(x,k);
     if (k==0)
       c = L2InnerProduct(pk,f,wC,a,b) / pi;
     else
       c = L2InnerProduct(pk,f,wC,a,b) * 2 / pi;
     end
     p = p + c*pk(x);
   end
   e = abs(f(x)-p);
   plot(x, p, 'DisplayName', sprintf('p_{%i}(x), error = %.2e',n,norm(e,inf)))
   figure(6)
   semilogy(x, e, 'DisplayName', sprintf('|f-p_{%i}|',n))

end

% finalize plots
figure(1)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Chebyshev interpolant')

figure(2)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Chebyshev interpolant error')

figure(3)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Legendre projection')

figure(4)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Legendre projection error')

figure(5)
hold off
xlabel('x')
ylabel('f(x), p(x)')
legend('Location','Northwest')
title('Chebyshev projection')

figure(6)
hold off
set(gca,'YScale','log')
xlabel('x')
ylabel('|f(x)-p(x)|')
legend('Location','Northwest')
title('Chebyshev projection error')

% end of script
