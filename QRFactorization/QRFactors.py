# QRFactors.py
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy

def QRFactors(A):
    """
    usage: Q, R = QRFactors(A)

    Function to compute the QR factorization of a (possibly rank-deficient)
    'thin' matrix A (m x n, with m >=n) using Householder reflection matrices.

    Input:    A - thin matrix
    Outputs:  Q - orthogonal matrix
              R - "upper triangular" matrix, i.e. R = [ Rhat ]
                                                      [  0   ]
                  with Rhat an (n x n) upper-triangular matrix
    """

    # get dimensions of A
    m, n = numpy.shape(A)

    # initialize results
    Q = numpy.identity(m)
    R = A.copy()

    # iterate over columns
    for k in range(n):

        # extract subvector from diagonal down and compute norm
        z = R[k:m,k]
        v = -z;
        v[0] = -numpy.sign(z[0])*numpy.linalg.norm(z) - z[0];
        vnorm = numpy.linalg.norm(v)

        # if subvector has norm zero, continue to next column
        if (vnorm < numpy.finfo(float).eps):
            continue

        # compute u = u = v/||v||;
        # the Householder matrix is then Qk = I-2*u*u'
        u = v/vnorm

        # update rows k through m of R
        for j in range(k,n):
            utR = 2 * u.T @ R[k:m, j]
            R[k:m, j] -= u*utR

        # update rows k through m of Q
        for j in range(m):
            utQ = 2 * u.T @ Q[k:m, j]
            Q[k:m, j] -= u*utQ

    # transpose Q before return
    Q = Q.T

    return [Q, R]

# end function
