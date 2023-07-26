from main import *
import RustyLab1
import timeit
from matplotlib import pyplot as plt
import datetime

results = np.zeros((34, 3))
results[:, 0] = np.linspace(2, 35, num=34) ** 2
tol = 1e-6
for i in range(34):
    A = np.random.random((int(results[i, 0]), int(results[i, 0])))
    results[i, 1] = timeit.timeit(lambda: np.matmul(A, A), number=10000)
    current_time = datetime.datetime.now()
    print("Finished:", i, "Numpy at", current_time)
    results[i, 2] = timeit.timeit(lambda: RustyLab1.matmul(A, A), number=10000)
    current_time = datetime.datetime.now()
    print("Finished:", i, "Rust at", current_time)


plt.title("Rust vs Numpy (Matmul Function)")
plt.xlabel("Matrix Size")
plt.ylabel("Time")
plt.plot(results[:, 0], results[:, 1], label="Numpy")
plt.plot(results[:, 0], results[:, 2], label="Rust")
plt.legend()
plt.show()

