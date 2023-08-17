#Numerical Approximations: Monte Carlo Quadrifolium Area and Airy Function

#Two numerical approximation methods:
#- The first method uses the Monte Carlo technique to approximate the area of the quadrifolium curve.
#- The second method calculates the Airy function using the trapezoidal rule for numerical integration.


#Monte Carlo Quadrifolium Area Approximation

import random
import math
import time

random.seed(123)

def monte_carlo_quadrifolium(num_samples):
    total_sum = 0

    for _ in range(num_samples):
        # Generate random theta and r within the range 0 to π/4 and 0 to 1 respectively.
        theta = random.uniform(0, math.pi/4)
        r = random.uniform(0, 1)

        sin_squared_2theta = (math.sin(2 * theta))**2

        total_sum += sin_squared_2theta

    average_sin_squared_2theta = total_sum / num_samples

    # one leaf
    area_one_leaf = (math.pi / 4) * average_sin_squared_2theta

    # four leaves.
    total_area = 4 * area_one_leaf

    return total_area

def main():
    # Set the number of random samples for Monte Carlo approximation.
    num_samples = 1000000

    start_time = time.time()

    result = monte_carlo_quadrifolium(num_samples)

    end_time = time.time()

    run_time = end_time - start_time

    print("Q2.1")
    print(f"Total area of the four leaves: {result:.4f}")
    print(f"Runtime: {run_time:.5f} seconds.")

if __name__ == "__main__":
    main()



#Numerical Approximation of Airy Function using Trapezoidal Rule
import numpy as np

def integrand(t, x):
    return np.cos(t**3/3 + x*t)

def trapezoidal_rule(a, b, n, x):
    h = (b - a) / n
    sum_result = (integrand(a, x) + integrand(b, x)) / 2

    for i in range(1, n):
        sum_result += integrand(a + i * h, x)

    return sum_result * h / np.pi

def airy_function(x):
    n = 5000  # Number of intervals for numerical integration
    upper_bound = 25  # Approximating infinity

    if x < -upper_bound or x > upper_bound:
        raise ValueError("The function is not defined for x outside the range (-25, 25)")

    result = trapezoidal_rule(0, upper_bound, n, x)
    return result

# Test
x_values = [-3, -2, -1, 0, 1, 2, 3]

print("\nQ2.2")
print("𝑥     Ai(𝑥) - Expected")
print("-------------------------")

for x in x_values:
    start_time = time.time()
    result = round(airy_function(x), 6)
    end_time = time.time()
    run_time = end_time - start_time

    print(f"{x}     {result} - {round(airy_function(x), 3)} (Runtime: {run_time:.6f} seconds)")
