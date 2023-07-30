from typing import Tuple, List

import numpy as np

try:
    import RustyLab1
    RustyLab1_found = True
except:
    RustyLab1_found = False


def power(A: np.ndarray, tol: float) -> Tuple[np.ndarray, float]:
    s = np.random.random(
        size=A.shape[1]
    )
    old_eigenvalue = 0.

    s = s / np.linalg.norm(s)
    converged = False
    while not converged:
        t = np.matmul(A, s)
        i = max(
            np.argmax(t),
            np.argmin(t),
            key=abs
        )
        eigenvalue = abs(np.linalg.norm(t))
        s = t / eigenvalue
        if abs(eigenvalue - old_eigenvalue) / abs(eigenvalue) > tol:
            old_eigenvalue = eigenvalue
        else:
            converged = True

    if np.matmul(A, s)[i] / s[i] < 0:
        eigenvalue = -eigenvalue

    return s, eigenvalue


def deflate(A: np.ndarray, eigenvector: np.ndarray, eigenvalue: float) -> np.ndarray:
    return A - eigenvalue * np.matmul(eigenvector[np.newaxis].T, eigenvector[np.newaxis])


def power_w_deflate(A: np.ndarray, tol: float) -> List[Tuple[np.ndarray, float]]:
    eigenpairs = []
    for i in range(A.shape[1]):
        eigenpairs.append(power(A, tol))
        A = deflate(A, *eigenpairs[i])

    return eigenpairs


def spring_system_init(K: np.ndarray) -> np.ndarray:
    system = np.diag(K)
    system_diag = np.diag(-K[1:None])
    system[1:None, 0:-1] = system[1:None, 0:-1] + system_diag
    system[0:-1, 1:None] = system[0:-1, 1:None] + system_diag
    system[0:-1, 0:-1] = system[0:-1, 0:-1] - system_diag
    return system


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    spring_sys = spring_system_init(np.ones(10))
    eig_pairs = power_w_deflate(spring_sys, 1e-6)
    nat_freq = [pair[1]**0.5 / 2*np.pi for pair in eig_pairs]

    plt.title("Eigenvalue vs Eigenvalue Index")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Natural Frequency")
    plt.plot(nat_freq)
    plt.xticks(range(0,10))

    plt.figure()
    plt.title("Eigenvector Value vs Eigenvector Value Index")
    plt.xlabel('Eigenvector Value Index')
    plt.ylabel('Eigenvector Value')
    [plt.plot(pair[0]) for pair in eig_pairs]
    plt.legend(range(10))

    plt.figure()
    fig, axs = plt.subplots(5,2)
    fig.suptitle("Eigenvector Value vs Eigenvector Value Index")
    for ax in axs.flat:
        ax.set(
            xlabel='Eigenvector Value Index'
        )
        ax.set_xticks(
            range(0, 10)
        )

    for i in range(0,5):
        axs[i, 0].plot(range(0, 10), eig_pairs[i][0])

    for i in range(0,5):
        axs[i, 1].plot(range(0, 10), eig_pairs[i+5][0])

    axs[2, 0].set(
        ylabel='Eigenvector Value'
    )

    # Question 3
    eig_val, eig_vec = np.linalg.eig(spring_sys)
    order = np.flip(eig_val.argsort())
    eig_val = eig_val[order]
    eig_val = [val**0.5 / 2*np.pi for val in eig_val]
    eig_vec = eig_vec[order]

    tolerances = np.logspace(-8, -2, num=4)
    solutions = [power_w_deflate(spring_sys, tol) for tol in tolerances]

    plt.figure()
    plt.title("Natural Frequencies vs Tolerance")
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Natural Frequencies')
    for sol in solutions:
        nat_freq = [pair[1]**0.5 / 2*np.pi for pair in eig_pairs]
        plt.plot(nat_freq)
    plt.plot(eig_val)

    plt.figure()
    plt.title("Natural Frequency Errors vs Tolerance")
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Error')
    for sol in solutions:
        nat_freq = [pair[1] ** 0.5 / 2 * np.pi for pair in sol]
        for i in range(10):
            nat_freq[i] = (nat_freq[i] - (eig_val[i])) / eig_val[i]

        plt.plot(nat_freq)
    plt.legend(np.logspace(-8, -2, num=4))

    plt.show()
    pass
