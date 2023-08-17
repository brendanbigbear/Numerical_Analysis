import time
import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation
def f(x, y, y_prime):
    return np.exp(-y_prime) - y_prime - (x**2 + y**2)

# Euler's method
def euler_method(f, x0, y0, y_prime0, h, N):
    x_values = [x0]
    y_values = [y0]

    for _ in range(N):
        y_prime = y_prime0 + h * f(x0, y0, y_prime0)
        y = y0 + h * y_prime
        x0 += h
        y0 = y
        y_prime0 = y_prime
        x_values.append(x0)
        y_values.append(y0)

    return x_values, y_values

# Runge-Kutta 4th order method
def runge_kutta_4th_order(f, x0, y0, y_prime0, h, N):
    x_values = [x0]
    y_values = [y0]

    for _ in range(N):
        k1 = h * f(x0, y0, y_prime0)
        k2 = h * f(x0 + h/2, y0 + h/2 * y_prime0 + k1/2, y_prime0 + k1/2)
        k3 = h * f(x0 + h/2, y0 + h/2 * y_prime0 + k2/2, y_prime0 + k2/2)
        k4 = h * f(x0 + h, y0 + h * y_prime0 + k3, y_prime0 + k3)

        y_prime = y_prime0 + (k1 + 2*k2 + 2*k3 + k4) / 6
        y = y0 + h * y_prime
        x0 += h
        y0 = y
        y_prime0 = y_prime
        x_values.append(x0)
        y_values.append(y0)

    return x_values, y_values

# Parameters
x0 = 0
y0 = -1
y_prime0 = 0
h = 1e-4
N_small = 10000
N_large = 100000

# Euler's method solutions
start_time = time.time()
euler_small_x, euler_small_y = euler_method(f, x0, y0, y_prime0, h, N_small)
end_time = time.time()
euler_small_runtime = end_time - start_time

start_time = time.time()
euler_large_x, euler_large_y = euler_method(f, x0, y0, y_prime0, h, N_large)
end_time = time.time()
euler_large_runtime = end_time - start_time


# Runge-Kutta 4th order method solutions
start_time = time.time()
rk4_small_x, rk4_small_y = runge_kutta_4th_order(f, x0, y0, y_prime0, h, N_small)
end_time = time.time()
rk4_small_runtime = end_time - start_time

start_time = time.time()
rk4_large_x, rk4_large_y = runge_kutta_4th_order(f, x0, y0, y_prime0, h, N_large)
end_time = time.time()
rk4_large_runtime = end_time - start_time


# Plot the solutions in a single grid
plt.figure(figsize=(10, 6))

plt.plot(euler_small_x, euler_small_y, label=f"Euler (N=10,000) - Runtime: {euler_small_runtime:.4f} s")
plt.plot(euler_large_x, euler_large_y, label=f"Euler (N=100,000) - Runtime: {euler_large_runtime:.4f} s")
plt.plot(rk4_small_x, rk4_small_y, label=f"Runge-Kutta 4 (N=10,000) - Runtime: {rk4_small_runtime:.4f} s")
plt.plot(rk4_large_x, rk4_large_y, label=f"Runge-Kutta 4 (N=100,000) - Runtime: {rk4_large_runtime:.4f} s")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Numerical Solutions of the Initial Value Problem with Runtimes")
plt.legend()
plt.grid()


plt.show()
