#Solving a Linear System and Computing Vector Products with Numpy
import time
import numpy as np

# Set matrix
N = 25

# Random N x N -5~5
A = np.random.uniform(-5, 5, size=(N, N))

# b
b = np.ones(N)

start_time = time.time()

# Solve
x = np.linalg.solve(A, b)

# Compute b1 = Ax
b1 = np.dot(A, x)
b1_deciaml = np.round(b1, 3)
b1_final = ["{:.3f}".format(value) for value in b1_deciaml]

end_time = time.time()

# Solution
print("Solution vector x:")
print(x)

print("\nComputed vector b1:")
print(b1_final)

runtime = end_time - start_time

print("\nRuntime:", runtime, "seconds")


