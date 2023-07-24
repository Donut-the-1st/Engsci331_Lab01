from typing import Tuple

import numpy as np
import RustyLab1
import timeit
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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


# A = np.array([[2., -1.], [-1., 2.]])
# A = np.random.rand(2, 2) + np.random.rand(2, 2)
# print(power(A, 1e-6))
# print(RustyLab1.power(A, 1e-6))
# print(np.linalg.eig(A))
#
# print("2x2 Matrix")
# print("Python: %3f" % timeit.timeit(lambda: power(A, 1e-6), number=10000))
# print("Rust: %3f" % timeit.timeit(lambda: RustyLab1.power(A, 1e-6), number=10000))
#
# A = np.random.rand(10, 10) + np.random.rand(10, 10)
# print("10x10 Matrix")
# print("Python: %3f" % timeit.timeit(lambda: power(A, 1e-6), number=10000))
# print("Rust: %3f" % timeit.timeit(lambda: RustyLab1.power(A, 1e-6), number=10000))
#
# A = np.random.rand(100, 100) + np.random.rand(100, 100)
# print("100x100 Matrix")
# print("Python: %3f" % timeit.timeit(lambda: power(A, 1e-6), number=10000))
# print("Rust: %3f" % timeit.timeit(lambda: RustyLab1.power(A, 1e-6), number=10000))
#
# A = np.random.rand(1000, 1000) + np.random.rand(1000, 1000)
# print("1000x1000 Matrix")
# print("Python: %3f" % timeit.timeit(lambda: power(A, 1e-6), number=10000))
# print("Rust: %3f" % timeit.timeit(lambda: RustyLab1.power(A, 1e-6), number=10000))
#
# A = np.random.rand(2000, 2000) + np.random.rand(2000, 2000)
# print("2000x2000 Matrix")
# print("Python: %3f" % timeit.timeit(lambda: power(A, 1e-6), number=10000))
# print("Rust: %3f" % timeit.timeit(lambda: RustyLab1.power(A, 1e-6), number=10000))

tests = 2000
results = np.concatenate((np.arange(2., tests + 2)[:, np.newaxis], np.zeros((tests, 2))), axis=1)
results_df = pd.DataFrame(
    results,
    columns=["Matrix Size", "Python", "Rust"]
)

for i in range(2, tests + 1):
    for j in range(20):
        A = np.random.rand(i, i) + np.random.rand(i, i)
        results[i - 1 , 1] = results[i - 1, 1] + timeit.timeit(lambda: power(A, 1e-6), number=5000)
        results[i - 1, 2] = results[i - 1, 2] + timeit.timeit(lambda: RustyLab1.power(A, 1e-6), number=5000)

sns.set_theme()
sns.lineplot(data=results_df[['Python', 'Rust']], palette=['green', 'orange']).set(title='Power Method', xlabel="Matrix Size", ylabel="Combined Time (100k tests)")

plt.show()
results_df.to_csv("results.csv")
