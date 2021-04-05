% Driver to test various ODE solvers for the stiff IVP system
%    [u1;u2]' = [-100, 1; -1, -100]*[u1;u2] + [100*sin(10*t); 100*cos(10*t)],   0<t<1
%    [u1;u2](0) = [10; 20].
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% ODE RHS and derivatives, initial condition, etc.
A   = [-100.0, 1.0; -1.0, -100.0];
f   = @(t,u) A*u + [100*sin(10*t); 100*cos(10*t)];
f_u = @(t,u) A;
u0  = [10; 20];
t0  = 0;
Tf  = 1;
hvals = 0.5.^(5:11);

% initialize error storage arrays
err_RK2    = zeros(size(hvals));
err_Heun   = zeros(size(hvals));
err_AB2    = zeros(size(hvals));
err_fE     = zeros(size(hvals));
err_bE     = zeros(size(hvals));
err_trap   = zeros(size(hvals));

% iterate over our h values
for j=1:length(hvals)

   % create time output array
   h = hvals(j);
   N = floor((Tf-t0)/h);
   tvals = linspace(t0, Tf, N+1);

   % construct reference solution
   odeopts = odeset('Jacobian', f_u, 'RelTol', 1e-6, 'AbsTol', 1e-10);
   [t_ref,u_ref] = ode15s(f, tvals, u0, odeopts);

   % initialize outputs
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
      err_fE(j) = max([err_fE(j), norm(u_fE'-u_ref(i+1,:))]);

      u_RK2 = RK2_step(f, t, u_RK2, h);
      err_RK2(j) = max([err_RK2(j), norm(u_RK2'-u_ref(i+1,:))]);

      u_Heun = Heun_step(f, t, u_Heun, h);
      err_Heun(j) = max([err_Heun(j), norm(u_Heun'-u_ref(i+1,:))]);

      % for first AB2 step, store fold and use RK2 for step; subsequently use AB2
      if (i == 1)
         fold_AB2 = f(t, u_AB2);
         u_AB2 = RK2_step(f, t, u_AB2, h);
      else
         [u_AB2,fold_AB2] = AB2_step(f, t, u_AB2, h, fold_AB2);
      end
      err_AB2(j) = max([err_AB2(j), norm(u_AB2'-u_ref(i+1,:))]);

      u_bE = bwd_Euler_step(f, f_u, t, u_bE, h);
      err_bE(j) = max([err_bE(j), norm(u_bE'-u_ref(i+1,:))]);

      u_trap = trap_step(f, f_u, t, u_trap, h);
      err_trap(j) = max([err_trap(j), norm(u_trap'-u_ref(i+1,:))]);

   end
end

% output convergence results
disp('Results for Fwd Euler:')
err = err_fE;
fprintf('   h = %12g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %12g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for RK2:')
err = err_RK2;
fprintf('   h = %12g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %12g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for Heun:')
err = err_Heun;
fprintf('   h = %12g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %12g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for AB2:')
err = err_AB2;
fprintf('   h = %10g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %12g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for Bwd Euler:')
err = err_bE;
fprintf('   h = %12g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %12g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

disp('Results for Trapezoidal:')
err = err_trap;
fprintf('   h = %12g,  err = %.2e\n',hvals(1),err(1))
for i=2:length(hvals)
   fprintf('   h = %12g,  err = %.2e,  rate = %g\n',...
           hvals(i),err(i),log(err(i)/err(i-1))/log(hvals(i)/hvals(i-1)))
end

% display reference solution
figure()
plot(t_ref, u_ref)
xlabel('t')
ylabel('u(t)')
legend('u_1(t)','u_2(t)')

% end of script
