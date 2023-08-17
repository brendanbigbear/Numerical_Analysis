#Numerical Root Finding Methods and Floating-Point Operation Analysis

#Implements various numerical root finding methods, including the Bisection Method,
# Newton's Method, Secant Method, and Fixed-Point Iterations. It estimates the roots of a given function
# and calculates the number of floating-point operations required for each method.

from math import sin, cos

# Function to estimate the number of floating-point operations for each method
def estimate_operations(method_function, *args):
    root, iterations, num_operations = method_function(*args)
    return root, iterations, num_operations

# Bisection Method
def g(x):
    return sin(x) + (x - 1) ** 3

def bisection_method(a, b, tolerance=0.5e-4):
    num_operations = 0
    iterations = 0
    while (b - a) / 2 > tolerance:
        c = (a + b) / 2
        if g(c) == 0:
            return c, iterations, num_operations
        elif g(a) * g(c) < 0:
            b = c
        else:
            a = c
        num_operations += 3  # Three basic arithmetic operations per iteration (addition, division, subtraction)
        iterations += 1
    root = (a + b) / 2
    return root, iterations, num_operations

# Newton's Method
def g(x):
    return sin(x) + (x - 1) ** 3

def g_prime(x):
    return cos(x) + 3 * (x - 1) ** 2

def newton_method(x0, tolerance=0.5e-4, max_iterations=100):
    num_operations = 0
    iterations = 0
    while True:
        g_prime_val = g_prime(x0)
        x1 = x0 - g(x0) / g_prime_val
        if abs(x1 - x0) < tolerance:
            return x1, iterations, num_operations
        x0 = x1
        num_operations += 4  # Four basic arithmetic operations per iteration (addition, division, subtraction, multiplication)
        iterations += 1
        if iterations == max_iterations:
            raise ValueError("Maximum iterations reached")

# Secant Method
def g(x):
    return sin(x) + (x - 1) ** 3

def secant_method(x0, x1, tolerance=0.5e-4, max_iterations=100):
    num_operations = 0
    iterations = 0
    while True:
        x2 = x1 - (g(x1) * (x1 - x0)) / (g(x1) - g(x0))
        if abs(x2 - x1) < tolerance:
            return x2, iterations, num_operations
        x0 = x1
        x1 = x2
        num_operations += 4  # Four basic arithmetic operations per iteration (addition, division, subtraction, multiplication)
        iterations += 1
        if iterations == max_iterations:
            raise ValueError("Maximum iterations reached")

# Fixed-Point Iterations
def g(x):
    return sin(x) + (x - 1) ** 3

def g_prime(x):
    return cos(x) + 3 * (x - 1) ** 2

def fixed_point_iterations(x0, tolerance=0.5e-4, max_iterations=100):
    num_operations = 0
    iterations = 0
    while True:
        x1 = x0 - g(x0) / g_prime(x0)
        if abs(x1 - x0) < tolerance:
            return x1, iterations, num_operations
        x0 = x1
        num_operations += 3  # Three basic arithmetic operations per iteration (addition, division, subtraction)
        iterations += 1
        if iterations == max_iterations:
            raise ValueError("Maximum iterations reached")

# Initial values
#for Bisection Method
a0_bisection = 0
b0_bisection = 1
#for Newton Method
x0_newton = 0.8
#Secant Method
x0_secant = 0
x1_secant = 1
#Fixed-Point Iterations
x0_fixed_point = 1

# Compute the number of iterations and floating-point operations for each method
root_bisection, iterations_bisection, num_operations_bisection = estimate_operations(bisection_method, a0_bisection, b0_bisection)
root_newton, iterations_newton, num_operations_newton = estimate_operations(newton_method, x0_newton)
root_secant, iterations_secant, num_operations_secant = estimate_operations(secant_method, x0_secant, x1_secant)
root_fixed_point, iterations_fixed_point, num_operations_fixed_point = estimate_operations(fixed_point_iterations, x0_fixed_point)

# Output the results
print("Bisection Method:")
print("Root:", root_bisection)
print("Iterations:", iterations_bisection)
print("Number of floating-point operations:", num_operations_bisection)

print("\nNewton's Method:")
print("Root:", root_newton)
print("Iterations:", iterations_newton)
print("Number of floating-point operations:", num_operations_newton)

print("\nSecant Method:")
print("Root:", root_secant)
print("Iterations:", iterations_secant)
print("Number of floating-point operations:", num_operations_secant)

print("\nFixed-Point Iterations:")
print("Root:", root_fixed_point)
print("Iterations:", iterations_fixed_point)
print("Number of floating-point operations:", num_operations_fixed_point)

#Polynomial Interpolation and Extrapolation

#Demonstrates polynomial interpolation and extrapolation using NumPy.
# Given a set of data points, it computes a polynomial of degree 5 that fits the data
# and then uses it to interpolate values at specific points. Additionally, it predicts the value
# of the polynomial at a new point.

import numpy as np

# Given data points
t_values = [6, 5, 4, 3, 2, 1]
y_values = [188.06, 185.27, 186.68, 187.00, 183.96, 185.01]

# Interpolate the data in a polynomial P5(t)
coefficients = np.polyfit(t_values, y_values, 5)
polynomial = np.poly1d(coefficients)

# Interpolate at t = 3.5, 5.5, and 1.5 for example points
t_interpolate_1 = 3.5
p5_t_interpolate_1 = polynomial(t_interpolate_1)

t_interpolate_2 = 5.5
p5_t_interpolate_2 = polynomial(t_interpolate_2)

t_interpolate_3 = 1.5
p5_t_interpolate_3 = polynomial(t_interpolate_3)

print("\nInterpolated value at t =", t_interpolate_1, "is:", p5_t_interpolate_1)
print("\nInterpolated value at t =", t_interpolate_2, "is:", p5_t_interpolate_2)
print("\nInterpolated value at t =", t_interpolate_3, "is:", p5_t_interpolate_3)

# Compute P5(t = 7) Predicting
t_interpolate_7 = 7
p5_t_interpolate_7 = polynomial(t_interpolate_7)

print("\nInterpolated value at t =", t_interpolate_7, "is:", p5_t_interpolate_7)