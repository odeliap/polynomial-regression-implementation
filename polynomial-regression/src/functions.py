# Libraries
import math
import numpy as np
import scipy.linalg as LA
# ------------------------------------------------------------------------------


# Linear / Polynomial Regression Functions
# ------------------------------------------------------------------------------

def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.

    Input:
       x: vector of input data.
       deg: degree of the polynomial fit.

    Output:
        A: polynomial regression matrix with shape (x, deg+1)

    """

    # check the formatting of the input array
    # we need to format it to a "matrix" type
    if x.ndim == 1:
        x = np.array(x)
        x = x[:, None]

    # initialize A as column of ones
    A = np.ones((len(x), 1))

    # loop through degrees and add them as columns
    for exp in range(1, deg + 1):
        col = x**exp
        A = np.hstack((col, A))

    return A


def LLS_Solve(x, y, deg):
    """Find the vector w that solves the least squares regression.

    Input:
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.

    Output:
       w: vector that solves least squares regression.

    """
    # get matrix A
    A = A_mat(x, deg)
    # get A transpose
    A_transpose = A.transpose()

    # get w = (A^T*A)^(-1)*A^T*y
    w = LA.inv(A_transpose.dot(A)).dot(A_transpose).dot(y)

    return w



def LLS_ridge(x, y, deg, lam):
    """Find the vector w that solves the ridge regression problem.

    Input:
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression.

    Output:
       w: vector w that solves the ridge regression problem

    """

    # get matrix A
    A = A_mat(x, deg)
    # get A transpose
    A_transpose = A.transpose()
    # get number of rows of matrix: A transpose times A
    # should be a square matrix
    matrix_len = len(A_transpose.dot(A))
    # make identity matrix of this size
    identity = np.identity(matrix_len)
    # get w
    w = LA.inv(A_transpose.dot(A) + (lam * identity)).dot(A_transpose).dot(y)
    return w


def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.

    Input:
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial.

    Output:
       output_data: vector of output data for the polynomial

    """

    # calculate degree  of polynomial by subtracting one from length of coeffs (w)
    degree = len(coeffs) - 1
    # get matrix A
    A = A_mat(data, degree)
    # multiply A with coeffs to get output data or polynomial
    output_data = A.dot(coeffs)
    return output_data


def LLS_func(x, y, w, deg):
    """The linear least squares objective function.

    Input:
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial.

    Output:
       objective_func: the objective function for the linear least squares problem

    """
    # get matrix A
    A = A_mat(x, deg)
    # get norm of A times w minus y
    norm = LA.norm(A.dot(w) - y)
    # get the objective function by squaring the norm
    objective_func = norm**2
    # return the objective function
    return objective_func


def RMSE(x, y, w):
    """Compute the root mean square error.

    Input:
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.

    Output:
       rmse: root mean square error

    """

    # calculate degree from length of w minus 1
    degree = len(w) - 1
    # get matrix A
    A = A_mat(x, degree)
    # get norm and square it
    norm_squared = LA.norm(y - A.dot(w))**2
    # get number of datapoints
    # N = x.size
    N = x.size
    # get one over N
    inv_N = 1/N
    # take square root of the norm squared to get root mean square error
    rmse = math.sqrt(inv_N * norm_squared)
    return rmse
