import numpy as np
def ThomasSolve(a, b, c, d) -> float:
    """
    Solves a tridiagonal system of linear equations using Thomas algorithm.

    Args:
        a (numpy.ndarray): Bottom diagonal
        b (numpy.ndarray): Main diagonal
        c (numpy.ndarray): Upper diagonal
        d (numpy.ndarray): Constant terms of linear equations
    Returns:
        numpy.ndarray: Solution of linear equations
    """
    n = len(b)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    c = np.insert(c, len(c), 0)
    a = np.insert(a, 0, 0)
    beta[0] = -c[0]/b[0]
    gamma[0] = d[0]/b[0]
    for i in range(1, n):
        beta[i] = -c[i]/(b[i] + a[i]*beta[i-1])
        gamma[i] = (d[i] - a[i]*gamma[i-1])/(b[i] + a[i]*beta[i-1])
    x = np.zeros(n)
    x[-1] = gamma[-1]
    for i in range(n-2, -1, -1):
        x[i] = gamma[i] + beta[i]*x[i+1]
    return x