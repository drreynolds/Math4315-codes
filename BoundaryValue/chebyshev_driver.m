% Demo to show Chebyshev collocation method for BVP
%     u'' + (1/(2+x)) u' + (11x/(2+x)) u = (-e^x (12x^3 + 7x^2 + 1))/(2+x), -1<x<1
%     u(-1) = u(1) = 0
% using various uniform mesh sizes.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear
close all

% set numbers of intervals for tests
nvals = [6, 8, 10, 12, 14];

% setup problem, analytical solution, etc
a = -1;
b = 1;
p = @(x) 1./(2+x);
q = @(x) 11*x./(2+x);
r = @(x) -exp(x).*(12*x.^3 + 7*x.^2 + 1)./(2+x);
utrue = @(x) exp(x).*(1-x.^2);

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
  [t,Dx] = differentiation_matrix(n, a, b, 0, 1);
  [t,Dxx] = differentiation_matrix(n, a, b, 0, 2);

  % create additional diagonal matrices
  P = diag(p(t));
  Q = diag(q(t));
  rhs = r(t);
  [A,rhs] = enforce_boundary(Dxx + P*Dx + Q,rhs);

  % solve linear system, compute error and h
  u = A\rhs;
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
title('Chebyshev Approximations')

figure(2)
set(gca,'YScale','log')
xlabel('x')
ylabel('error(x)')
legend()
title('Chebyshev Approximations')




%%% utility functions %%%

function [A,rhs] = enforce_boundary(A,rhs)
  % Utility routine to enforce the homogeneous Dirichlet boundary conditions on
  % the linear system encoded in the matrix A and right-hand side vector r.
  % This routine assumes that the inputs include placeholder rows for these
  % conditions.
  A(1,:) = 0*A(1,:);
  A(end,:) = 0*A(end,:);
  A(1,1) = max(abs(diag(A)));
  A(end,end) = max(abs(diag(A)));
  rhs(1) = 0;
  rhs(end) = 0;
end


% end of function
