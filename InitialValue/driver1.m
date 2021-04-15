% Driver to test various ODE solvers for the test problem
%    u' = (u+t^2-2)/(t+1),   t in [0,5]
%    u(0) = 2.
%
% Daniel R. Reynolds
% SMU Mathematics
% MATH 4315
clear

% ODE RHS and derivatives, initial condition, etc.
f   = @(t,u) (u+t.^2-2)./(t+1);
f_u = @(t,u) 1/(t+1);
utrue = @(t) t.^2 + 2*t + 2 - 2*(t+1).*log(t+1);
u0  = 2;
t0  = 0;
Tf  = 5;
hvals = 0.5.^(2:8);

% initialize error storage arrays
err_fE     = zeros(size(hvals));
err_RK2    = zeros(size(hvals));
err_Heun   = zeros(size(hvals));
err_AB2    = zeros(size(hvals));
err_bE     = zeros(size(hvals));
err_trap   = zeros(size(hvals));

% iterate over our h values
for j=1:length(hvals)

   % create time output array
   h = hvals(j);
   N = floor((Tf-t0)/h);
   tvals = linspace(t0,Tf,N+1);

   % initialize solutions
   u_fE     = u0;
   u_RK2    = u0;
   u_Heun   = u0;
   u_AB2    = u0;
   u_bE     = u0;
   u_trap   = u0;

   % try out our methods
   for i=1:N

      t = tvals(i);
      tnew = t + h;

      u_fE = fwd_Euler_step(f, t, u_fE, h);
      err_fE(j) = max([err_fE(j), abs(u_fE-utrue(tnew))]);

      u_RK2 = RK2_step(f, t, u_RK2, h);
      err_RK2(j) = max([err_RK2(j), abs(u_RK2-utrue(tnew))]);

      u_Heun = Heun_step(f, t, u_Heun, h);
      err_Heun(j) = max([err_Heun(j), abs(u_Heun-utrue(tnew))]);

      % for first AB2 step, store fold and use RK2 for step; subsequently use AB2
      if (i == 1)
         fold_AB2 = f(t, u_AB2);
         u_AB2 = RK2_step(f, t, u_AB2, h);
      else
         [u_AB2,fold_AB2] = AB2_step(f, t, u_AB2, h, fold_AB2);
      end
      err_AB2(j) = max([err_AB2(j), abs(u_AB2-utrue(tnew))]);

      u_bE = bwd_Euler_step(f, f_u, t, u_bE, h);
      err_bE(j) = max([err_bE(j), abs(u_bE-utrue(tnew))]);

      u_trap = trap_step(f, f_u, t, u_trap, h);
      err_trap(j) = max([err_trap(j), abs(u_trap-utrue(tnew))]);

   end
end

% output convergence results
disp('Results for Fwd Euler:')
err = err_fE;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %10g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for RK2:')
err = err_RK2;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %10g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for Heun:')
err = err_Heun;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %10g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for AB2:')
err = err_AB2;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %10g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for Bwd Euler:')
err = err_bE;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %10g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for Trapezoidal:')
err = err_trap;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %10g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

% display true solution
figure()
plot(tvals, utrue(tvals))
xlabel('t')
ylabel('u(t)')

% end of script
