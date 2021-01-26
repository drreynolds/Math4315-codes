% Script to test QRFactors on a variety of matrices.
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315
clear

% set matrix sizes for tests
nvals = [50, 100, 200, 400];

% full-rank square matrix tests
for n = nvals

   fprintf('Testing with full-rank square matrix of dimension %i\n',n);

   % create the matrix
   I = eye(n);
   A = rand(n,n) + I;

   % call QRFactors
   [Q,R] = QRFactors(A);

   % output results
   fprintf('   ||I-Q^TQ||     = %g\n', norm(I-Q'*Q,2));
   fprintf('   ||I-QQ^T||     = %g\n', norm(I-Q*Q',2));
   fprintf('   ||A-QR||       = %g\n', norm(A-Q*R,2));
   fprintf('   ||tril(R,-1)|| = %g\n', norm(tril(R,-1),2));

end

% full-rank thin matrix tests
for n = nvals

   fprintf('Testing with full-rank rectangular matrix of dimension %ix%i\n',2*n,n);

   % create the matrix
   I = eye(2*n);
   A = rand(2*n,n) + I(:,1:n);

   % call QRFactors
   [Q,R] = QRFactors(A);

   % output results
   fprintf('   ||I-Q^TQ||     = %g\n', norm(I-Q'*Q,2));
   fprintf('   ||I-QQ^T||     = %g\n', norm(I-Q*Q',2));
   fprintf('   ||A-QR||       = %g\n', norm(A-Q*R,2));
   fprintf('   ||tril(R,-1)|| = %g\n', norm(tril(R,-1),2));

end

% rank-deficient square matrix tests
for n = nvals

   fprintf('Testing with rank-deficient square matrix of dimension %i\n',n);

   % create the matrix
   I = eye(n);
   A = rand(n,n) + I;
   A(:,3) = 2*A(:,2);

   % call QRFactors
   [Q,R] = QRFactors(A);

   % output results
   fprintf('   ||I-Q^TQ||     = %g\n', norm(I-Q'*Q,2));
   fprintf('   ||I-QQ^T||     = %g\n', norm(I-Q*Q',2));
   fprintf('   ||A-QR||       = %g\n', norm(A-Q*R,2));
   fprintf('   ||tril(R,-1)|| = %g\n', norm(tril(R,-1),2));

end

% rank-deficient thin matrix tests
for n = nvals

   fprintf('Testing with rank-deficient rectangular matrix of dimension %ix%i\n',2*n,n);

   % create the matrix
   I = eye(2*n);
   A = rand(2*n,n) + I(:,1:n);
   A(:,3) = 2*A(:,2);

   % call QRFactors
   [Q,R] = QRFactors(A);

   % output results
   fprintf('   ||I-Q^TQ||     = %g\n', norm(I-Q'*Q,2));
   fprintf('   ||I-QQ^T||     = %g\n', norm(I-Q*Q',2));
   fprintf('   ||A-QR||       = %g\n', norm(A-Q*R,2));
   fprintf('   ||tril(R,-1)|| = %g\n', norm(tril(R,-1),2));

end
