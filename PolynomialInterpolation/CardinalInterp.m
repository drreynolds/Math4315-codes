% Script to plot cardinal functions for polynomial interpolants using various choices of node types.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% remove all existing variables
clear

% set numbers of nodes for tests
nvals = [5, 10, 15, 20];

% set interval that we will interpolate over
a = -1;
b = 1;

% set evaluation points to compare interpolants
x = linspace(a, b, 2001);

% loop over node numbers
for n = nvals

   fprintf('Plotting cardinal functions for n = %i\n', n);

   % set (n+1)x(n+1) identity matrix
   I = eye(n+1);

   % create figure window for this n
   figure()

   % uniformly-spaced nodes
   subplot(1,3,1)
   t = linspace(a,b,n+1);
   for i=1:n+1
     p = Lagrange(t, I(i,:), x);
     plot(x, p),
     hold on
   end
   xlabel('x')
   ylabel('I(e_{k})')
   title(sprintf('Uniformly spaced nodes, n = %i', n))

   % random nodes
   subplot(1,3,2)
   t = a + (b-a)*rand(1,n+1);
   for i=1:n+1
     p = Lagrange(t, I(i,:), x);
     plot(x, p),
     hold on
   end
   xlabel('x')
   ylabel('I(e_{k})')
   title(sprintf('Random nodes, n = %i', n))

   % Chebyshev nodes
   subplot(1,3,3)
   t = (a+b)/2 - (b-a)/2*cos((0:n)*pi/n);
   for i=1:n+1
     p = Lagrange(t, I(i,:), x);
     plot(x, p),
     hold on
   end
   xlabel('x')
   ylabel('I(e_{k})')
   title(sprintf('Chebyshev nodes, n = %i', n))

end


% end of script
