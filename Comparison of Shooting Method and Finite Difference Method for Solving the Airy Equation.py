#Comparison of Shooting Method and Finite Difference Method for Solving the Airy Equation
import time
import numpy as np
from scipy.integrate import solve_ivp

# Airy equation
def airy_eq(x, y):
    return [y[1], x * y[0]]

# Shooting method (s!=0)
def shooting_method(s, target, x_mesh):
    y_initial = [0, s]
    solution_s = solve_ivp(airy_eq, [x_mesh[0], x_mesh[-1]], y_initial)
    return solution_s.y[0, -1] - target

# Given
target_value = 17.76
x_mesh = np.linspace(0, 2, 9)  # Dividing 8
h = x_mesh[1] - x_mesh[0]

# Initial (shooting method)
s_lower = 0.0
s_upper = 5.0
tolerance = 1e-6
max_iterations = 20

start_time = time.time()

# Shooting method
for _ in range(max_iterations):
    s_mid = (s_lower + s_upper) / 2
    result_lower = shooting_method(s_lower, target_value, x_mesh)
    result_mid = shooting_method(s_mid, target_value, x_mesh)

    if result_mid * result_lower > 0:
        s_lower = s_mid
    else:
        s_upper = s_mid

    if abs(result_mid) < tolerance:
        break
end_time = time.time()

# Given(finite difference method)
a = 0.0
b = 2.0
N = 9
target_value = 17.76

# Mesh
h = (b - a) / (N - 1)
x_mesh = np.linspace(a, b, N)

# Initialize arrays
A = np.zeros((N, N))
b = np.zeros(N)

# Matrix A, Vector b
A[0, 0] = 1.0
A[N - 1, N - 1] = 1.0
b[N - 1] = target_value

for i in range(1, N - 1):
    A[i, i - 1] = 1 / h**2
    A[i, i] = -2 / h**2 - x_mesh[i]
    A[i, i + 1] = 1 / h**2

# Solution(finite difference method)
start_time_fd = time.time()
solution_fd = np.linalg.solve(A, b)
end_time_fd = time.time()

# Print
print("Shooting Method Solution:")
for i, (x, y) in enumerate(zip(x_mesh, solve_ivp(airy_eq, [x_mesh[0], x_mesh[-1]], [0, s_mid], t_eval=x_mesh).y[0])):
    print(f"x = {x:.2f}, y = {y:.4f}")
print("Shooting Method Runtime:", end_time - start_time, "seconds")

print("\nFinite Difference Method Solution:")
for i, (x, y) in enumerate(zip(x_mesh, solution_fd)):
    print(f"x = {x:.2f}, y = {y:.4f}")
print("Finite Difference Method Runtime:", end_time_fd - start_time_fd, "seconds")
