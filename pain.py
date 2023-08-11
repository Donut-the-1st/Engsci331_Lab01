from main import *
import RustyLab1
import timeit
from matplotlib import pyplot as plt
import datetime

num_stuff = 39

results = np.zeros((num_stuff, 3))
results[:, 0] = np.linspace(2, num_stuff+1, num=num_stuff) ** 2
tol = 1e-6
for i in range(num_stuff):
    A = np.random.random((int(results[i, 0]), int(results[i, 0])))
    results[i, 1] = timeit.timeit(lambda: power(A, tol), number=10000)
    current_time = datetime.datetime.now()
    print("Finished:", i, "Numpy at", current_time)
    results[i, 2] = timeit.timeit(lambda: RustyLab1.power(A, tol), number=10000)
    current_time = datetime.datetime.now()
    print("Finished:", i, "Rust at", current_time)


plt.title("Rust vs Numpy (Power Function)")
plt.xlabel("Matrix Size")
plt.ylabel("Time")
plt.plot(results[:, 0], results[:, 1], label="Numpy")
plt.plot(results[:, 0], results[:, 2], label="Rust")
plt.legend()
plt.show()

