function [Q,R] = QRFactors(A)
% usage: [Q,R] = QRFactors(A)
%
% Function to compute the QR factorization of a (possibly rank-deficient)
% 'thin' matrix A (m x n, with m >=n) using Householder reflection matrices.
%
% Input:    A - thin matrix
% Outputs:  Q - orthogonal matrix
%           R - essentially upper triangular matrix, i.e. R = [ Rhat ]
%                                                             [  0   ]
%               with Rhat an (n x n) upper-triangular matrix
%
% Daniel R. Reynolds
% SMU Mathematics
% Math 4315

% get dimensions of A
[m,n] = size(A);

% initialize results
Q = eye(m);
R = A;

% iterate over columns
for k=1:n

  % extract subvector from diagonal down and compute norm
  z = R(k:m,k);
  v = -z;
  v(1) = -sign(z(1))*norm(z) - z(1);
  vnorm = norm(v);

  % if subvector has norm zero, continue to next column
  if (vnorm < eps)
    continue
  end

  % compute u = v/||v||;
  % the Householder matrix is then Qk = I-2*u*u'
  u = v/vnorm;

  % update rows k through m of R
  for j=1:n
    utR = 2*u'*R(k:m,j);
    R(k:m,j) = R(k:m,j) - u*utR;
  end

  % update rows k through m of Q
  for j=1:m
    utQ = 2*u'*Q(k:m,j);
    Q(k:m,j) = Q(k:m,j) - u*utQ;
  end

end

% transpose Q before return
Q = Q';

% end function
