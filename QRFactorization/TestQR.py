#!/usr/bin/env python3
#
# Script to test QRFactors on a variety of matrices.
#
# Daniel R. Reynolds
# SMU Mathematics
# Math 4315

# imports
import numpy
from QRFactors import QRFactors

# set matrix sizes for tests
nvals = [50, 100, 200, 400]

# full-rank square matrix tests
for n in nvals:

    print("Testing with full-rank square matrix of dimension ", n)

    # create the matrix
    I = numpy.eye(n)
    A = numpy.random.rand(n,n) + I

    # call QRFactors
    Q, R = QRFactors(A)

    # output results
    print("   ||I-Q^TQ||     = ", numpy.linalg.norm(I-Q.T@Q,2))
    print("   ||I-QQ^T||     = ", numpy.linalg.norm(I-Q@Q.T,2))
    print("   ||A-QR||       = ", numpy.linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", numpy.linalg.norm(numpy.tril(R,-1),2))

# full-rank rectangular matrix tests
for n in nvals:

    print("Testing with full-rank rectangular matrix of dimension ", 2*n, "x", n)

    # create the matrix
    I = numpy.eye(2*n)
    A = numpy.random.rand(2*n,n) + I[:,:n]

    # call QRFactors
    Q, R = QRFactors(A)

    # output results
    print("   ||I-Q^TQ||     = ", numpy.linalg.norm(I-Q.T@Q,2))
    print("   ||I-QQ^T||     = ", numpy.linalg.norm(I-Q@Q.T,2))
    print("   ||A-QR||       = ", numpy.linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", numpy.linalg.norm(numpy.tril(R,-1),2))

# rank-deficient square matrix tests
for n in nvals:

    print("Testing with rank-deficient square matrix of dimension ", n)

    # create the matrix
    I = numpy.eye(n)
    A = numpy.random.rand(n,n) + I
    A[:,2] = 2*A[:,1]

    # call QRFactors
    Q, R = QRFactors(A)

    # output results
    print("   ||I-Q^TQ||     = ", numpy.linalg.norm(I-Q.T@Q,2))
    print("   ||I-QQ^T||     = ", numpy.linalg.norm(I-Q@Q.T,2))
    print("   ||A-QR||       = ", numpy.linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", numpy.linalg.norm(numpy.tril(R,-1),2))

# rank-deficient rectangular matrix tests
for n in nvals:

    print("Testing with rank-deficient rectangular matrix of dimension ", 2*n, "x", n)

    # create the matrix
    I = numpy.eye(2*n)
    A = numpy.random.rand(2*n,n) + I[:,:n]

    # call QRFactors
    Q, R = QRFactors(A)

    # output results
    print("   ||I-Q^TQ||     = ", numpy.linalg.norm(I-Q.T@Q,2))
    print("   ||I-QQ^T||     = ", numpy.linalg.norm(I-Q@Q.T,2))
    print("   ||A-QR||       = ", numpy.linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", numpy.linalg.norm(numpy.tril(R,-1),2))


# end of script
