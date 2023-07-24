from typing import Tuple

import numpy as np


def power(A: np.ndarray, tol: float) -> Tuple[np.ndarray, float]:
    s = np.random.random(
        size=A.shape[1]
    )
    old_eigenvalue = 0.

    s = s / np.linalg.norm(s)
    converged = False
    while not converged:
        t = np.matmul(A, s)
        i = np.argmax(t)
        eigenvalue = np.linalg.norm(t)
        s = t / abs(eigenvalue)
        if abs(eigenvalue - old_eigenvalue) / abs(eigenvalue) > tol:
            old_eigenvalue = eigenvalue
        else:
            converged = True

    if np.matmul(A, s)[i] / s[i] < 0:
        eigenvalue = -eigenvalue

    return s, eigenvalue

if __name__ == '__main__':
    A = np.array([[2., -1.], [-1., 2.]])
    print(power(A, 1e-6))
