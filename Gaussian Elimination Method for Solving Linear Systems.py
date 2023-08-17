#Gaussian Elimination Method for Solving Linear Systems

import time
import numpy as np

# Generate A
n = 64
A = np.diag(np.full(n, 40)) + np.random.uniform(-1, 1, (n, n))

#b
b = np.ones(n)

#Gaussian elimination method
def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        pivot_row = i
        for j in range(i + 1, n):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[pivot_row, i]):
                pivot_row = j
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:n])) / augmented_matrix[i, i]

    return x

start_time = time.time()

solution = gaussian_elimination(A, b)

verification_result = np.dot(A, solution)

end_time = time.time()

np.set_printoptions(precision=6, suppress=True, formatter={'all': lambda x: f"{x:.6f}"})

#Results
print("\nSolution x:")
print(solution)
print("\nVerification result (A * x):")
print(verification_result)
print("\nb:")
print(b)

runtime = end_time - start_time

print("\nRuntime:", runtime, "seconds")
