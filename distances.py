import numpy as np


def euclidean_distance(X, Y):
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be of type `ndarray`.")
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be of type `ndarray`.")
    if len(X.shape) != 2:
        raise ValueError("X must be 2D `ndarray`.")
    if len(Y.shape) != 2:
        raise ValueError("Y must be 2D `ndarray`.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("The 2nd dimension of X and Y must be the same size.")
    return np.sqrt(
        np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1) - 2 * X @ Y.T
    )


def cosine_distance(X, Y):
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be of type `ndarray`.")
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be of type `ndarray`.")
    if len(X.shape) != 2:
        raise ValueError("X must be 2D `ndarray`.")
    if len(Y.shape) != 2:
        raise ValueError("Y must be 2D `ndarray`.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("The 2nd dimension of X and Y must be the same size.")
    X_norms = np.linalg.norm(X, axis=1).reshape(-1, 1)
    Y_norms = np.linalg.norm(Y, axis=1).reshape(1, -1)
    return 1 - (X @ Y.T) / (X_norms @ Y_norms)

