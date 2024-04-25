import numpy as np

def cheb(N):
    """
    Computes the Chebyshev differentiation matix
    following the approach by Trefethen but with
    some modifications
    """
    t = np.linspace(0, 1, N)
    x = np.cos(np.pi * t)

    c = np.ones(N)
    c[0] = 2
    c[N-1] = 2

    c *= (-1) ** np.arange(N)

    X = np.tile(x, (N, 1)).T

    dX = X - X.T

    D = np.outer(c, 1 / c) / (dX + np.eye(N))
    D -= np.diag(np.sum(D.T, axis=0))

    # my adjustment
    D *= -1
    x = np.flipud(x)

    return (D, x)
