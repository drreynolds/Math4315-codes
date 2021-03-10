% Script to verify polynomial bases and inner product routine.
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

% quick test to verify orthogonality of first 41 Chebyshev polynomials
fprintf('Testing orthogonality of first 41 Chebyshev polynomials:\n');
passed = true;
for i = 0:40
  for j=i+1:40
    fi = @(x) Chebyshev(x,i);
    fj = @(x) Chebyshev(x,j);
    v = L2InnerProduct(fi,fj,wC,a,b);
    if (abs(v) > 1e-6)
      fprintf('  <p%i,p%i> = %e\n', i, j, v);
      passed = false;
    end
  end
end
if (passed)
  fprintf('  Tests passed\n');
end

% quick test to verify orthogonality of first 41 Legendre polynomials
fprintf('Testing orthogonality of first 41 Legendre polynomials:\n');
passed = true;
for i = 0:40
  for j = i+1:40
    fi = @(x) Legendre(x,i);
    fj = @(x) Legendre(x,j);
    v = L2InnerProduct(fi,fj,wL,a,b);
    if (abs(v) > 1e-6)
      fprintf('  <p%i,p%i> = %e\n', i, j, v);
      passed = false;
    end
  end
end
if (passed)
  fprintf('  Tests passed\n');
end

% quick test to verify norms for first 41 Chebyshev polynomials
fprintf('Testing norms for first 41 Chebyshev polynomials:\n');
passed = true;
for i = 0:40
  fi = @(x) Chebyshev(x,i);
  v = L2InnerProduct(fi,fi,wC,a,b);
  if (i==0)
    if (abs(v - pi) > 1e-6)
      fprintf('  <p0,p0> = %e\n', v);
      passed = false;
    end
  else
    if (abs(v - pi/2) > 1e-6)
      fprintf('  <p%i,p%i> = %e\n', i, i, v);
      passed = false;
    end
  end
end
if (passed)
  fprintf('  Tests passed\n');
end

% quick test to verify norms for first 41 Legendre polynomials
fprintf('Testing norms for first 41 Legendre polynomials:\n');
passed = true;
for i = 0:40
  fi = @(x) Legendre(x,i);
  v = L2InnerProduct(fi,fi,wL,a,b);
  if (abs(v - 1/(i+1/2)) > 1e-6)
      fprintf('  <p%i,p%i> = %e\n', i, i, v);
      passed = false;
  end
end
if (passed)
  fprintf('  Tests passed\n');
end

% end of script
