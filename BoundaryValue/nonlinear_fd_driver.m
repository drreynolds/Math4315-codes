% Demo to show second-order collocation method for the nonlinear BVP
%     u'' = 3(u')^2/u, -1<x<2
%     u(-1) = 1,  u(2) = 1/2
% using various uniform mesh sizes.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear
close all

% set numbers of intervals for tests
nvals = [10, 20, 40, 80, 160, 320];

% setup problem, analytical solution, etc
a = -1;
b = 2;
alpha = 1;
beta = 1/2;
utrue = @(x) 1./sqrt(x+2);

% initialize plots
figure(1)
x = linspace(a,b,1000)';
plot(x, utrue(x), 'DisplayName', 'u_{true}(x)')
hold on
figure(2)
hold on

% initialize 'current' mesh size and error norm
h_cur = 1;
e_cur = 1;

% loop over partition sizes
for n = nvals

  % get partition and differentiation matrices
  [t,Dx] = differentiation_matrix(n, a, b, 2, 1);
  [t,Dxx] = differentiation_matrix(n, a, b, 2, 2);

  % set nonlinear root-finding function and Jacobian
  phi = @(u) 3*((Dx*u).^2)./u;
  Jphi = @(u) 6*diag((Dx*u)./u)*Dx - 3*diag(((Dx*u)./u).^2);
  E = [zeros(n-1,1), eye(n-1), zeros(n-1,1)];
  f = @(u) [u(1)-alpha; E*(Dxx*u-phi(u)); u(end)-beta];
  J = @(u) [ 1, zeros(1,n) ; E*(Dxx - Jphi(u)); zeros(1,n), 1];

  % set initial condition to satisfy boundary conditions
  u = 1 - 1/6*(linspace(-1,2,n+1)'+1);

  % call Newton method to solve root-finding problem
  fprintf('Calling Newton method to solve for mesh size n = %i:\n', n);
  [u, its] = newton(f, J, u, 20, 1e-9, 1e-11, 1);

  % compute error and h
  e = abs(u-utrue(t));
  e_old = e_cur;
  e_cur = norm(e,'inf');
  h_old = h_cur;
  h_cur = max(t(2:end)-t(1:end-1));

  % update plots
  figure(1)
  plot(t, u, 'DisplayName', sprintf('u_{%i}(x)',n))
  figure(2)
  semilogy(t, e, 'DisplayName', sprintf('|u_{true}-u_{%i}|',n))

  % output current error norm and estimated convergence rate
  if (n > nvals(1))
    fprintf('n = %3i,  h = %.2e, ||error|| = %.2e,  conv. rate = %.2f\n', n, h_cur, e_cur, ...
           log(e_cur/e_old)/log(h_cur/h_old))
  else
    fprintf('n = %3i,  h = %.2e, ||error|| = %.2e\n', n, h_cur, e_cur)
  end

end

% finalize plots
figure(1)
xlabel('x')
ylabel('u(x)')
legend()
title('Second-order Nonlinear FD Approximations')

figure(2)
set(gca,'YScale','log')
xlabel('x')
ylabel('error(x)')
legend()
title('Second-order Nonlinear FD Error')


% end of function
