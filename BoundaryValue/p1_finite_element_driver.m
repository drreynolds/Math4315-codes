% Demo to show piecewise linear finite element solver for BVP
%     [(2+x) u']' + 11xu = -e^x (12x^3 + 7x^2 + 1), -1<x<1
%     u(-1)=u(1)=0
% using various non-uniform meshes.
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
b = 1;
c = @(x) -(2+x);
s = @(x) 11*x;
f = @(x) -exp(x).*(12*x.^3 + 7*x.^2 + 1);
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

  % get partition
  t = partition(a,b,n);

  % construct linear system
  K = stiffness_matrix(c, t);
  M = mass_matrix(s, t);
  r = rhs_vector(f, t);
  [A,r] = enforce_boundary(K+M,r);

  % solve linear system, compute error and h
  u = A\r;
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
title('P1 FEM Approximations')

figure(2)
set(gca,'YScale','log')
xlabel('x')
ylabel('error(x)')
legend()
title('P1 FEM Approximation Error')




%%% utility functions %%%

function t = partition(a, b, n)
  % Utility routine to construct the non-uniform partition, t, for the
  % interval [a,b] using (n+1) nodes.  For convenience I just use the
  % Chebyshev nodes of the second kind.  Returns t as a column vector.
  t = (a+b)/2 - (b-a)/2*cos((0:n)*pi/n);
  t = t';
end

function K = stiffness_matrix(c, t)
  % Utility routine to construct the stiffness matrix, K, based on the
  % coefficient function c(x) and partition t.  This matrix includes rows
  % for the boundary nodes.
  Ke = [1, -1; -1, 1];
  n = length(t)-1;
  h = t(2:n+1)-t(1:n);
  cvals = c(t);
  cavg = 0.5*(cvals(2:n+1)+cvals(1:n));
  K = zeros(n+1,n+1);
  K(2,2) = cavg(1)/h(1);
  K(n,n) = cavg(n)/h(n);
  for i=3:n
    K(i-1:i,i-1:i) = K(i-1:i,i-1:i) + (cavg(i-1)/h(i-1))*Ke;
  end
end

function M = mass_matrix(s, t)
  % Utility routine to construct the mass matrix, M, based on the
  % coefficient function s(x) and partition t.  This matrix includes rows
  % for the boundary nodes.
  Me = (1/6)*[2, 1; 1, 2];
  n = length(t)-1;
  h = t(2:n+1)-t(1:n);
  svals = s(t);
  savg = 0.5*(svals(2:n+1)+svals(1:n));
  M = zeros(n+1,n+1);
  M(2,2) = savg(1)*h(1)/3;
  M(n,n) = savg(n)*h(n)/3;
  for i=3:n
    M(i-1:i,i-1:i) = M(i-1:i,i-1:i) + (savg(i-1)*h(i-1))*Me;
  end
end

function r = rhs_vector(f, t)
  % Utility routine to construct the right-hand side vector, b, based on the
  % forcing function f(x) and partition t.  This vector includes entries
  % for the boundary nodes.
  fe = (1/2)*[1; 1];
  n = length(t)-1;
  h = t(2:n+1)-t(1:n);
  fvals = f(t);
  favg = 0.5*(fvals(2:n+1)+fvals(1:n));
  r = zeros(n+1,1);
  r(2) = favg(1)*h(1)/2;
  r(n) = favg(n)*h(n)/2;
  for i=3:n
    r(i-1:i) = r(i-1:i) + (favg(i-1)*h(i-1))*fe;
  end
end

function [A,r] = enforce_boundary(A,r)
  % Utility routine to enforce the homogeneous Dirichlet boundary conditions on
  % the linear system encoded in the matrix A and right-hand side vector r.
  % This routine assumes that the inputs include placeholder rows for these
  % conditions.
  A(1,:) = 0*A(1,:);
  A(end,:) = 0*A(end,:);
  A(1,1) = max(abs(diag(A)));
  A(end,end) = max(abs(diag(A)));
  r(1) = 0;
  r(end) = 0;
end


% end of function
