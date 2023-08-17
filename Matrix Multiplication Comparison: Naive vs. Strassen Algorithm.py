#Matrix Multiplication Comparison: Naive vs. Strassen Algorithm


import time
class RandomGenerator:
    def __init__(self, seed=123, a=1364525, c=1013904823, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return (self.state / self.m) * 2 - 1

def generate_random_matrix(rows, cols, generator):
    matrix = []
    for _ in range(rows):
        row = [generator.random() for _ in range(cols)]
        matrix.append(row)
    return matrix

def save_matrix_to_file(matrix, filename):
    with open(filename, "w") as f:
        for row in matrix:
            f.write(" ".join(f"{x:.4f}" for x in row))
            f.write("\n")

def multiply_matrices_naive(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must match the number of rows in the second matrix")

    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    multiplications = 0
    additions = 0

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
                multiplications += 1
                additions += 1

    return result, multiplications, additions

def load_matrix_from_file(filename):
    matrix = []
    with open(filename, "r") as f:
        for line in f:
            row = [float(x) for x in line.split()]
            matrix.append(row)
    return matrix

start_time = time.time()

rows = 512
cols = 512

generator = RandomGenerator()

matrix1 = generate_random_matrix(rows, cols, generator)
matrix2 = generate_random_matrix(rows, cols, generator)

save_matrix_to_file(matrix1, "matrix1(512).txt")
save_matrix_to_file(matrix2, "matrix2(512).txt")

print("Matrices saved to 'matrix1(512).txt' and 'matrix2(512).txt'.")

result_matrix, num_multiplications, num_additions = multiply_matrices_naive(matrix1, matrix2)

save_matrix_to_file(result_matrix, "multiplication_result(512).txt")

end_time = time.time()
run_time = end_time - start_time

print("\nNumber of multiplications:", num_multiplications)
print("Number of additions:", num_additions)

print("\nrun_time:", run_time, "seconds")

def add_matrices(matrix1, matrix2):
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] + matrix2[i][j])
        result.append(row)
    return result

def subtract_matrices(matrix1, matrix2):
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    return result

def strassen_multiply(matrix1, matrix2, threshold, level=1):
    if len(matrix1) <= threshold or level >= 3:
        result_matrix, multiplications, additions = multiply_matrices_naive(matrix1, matrix2)
        return result_matrix, multiplications, additions

    size = len(matrix1)
    half_size = size // 2

    multiplications = 0
    additions = 0

    a11 = [row[:half_size] for row in matrix1[:half_size]]
    a12 = [row[half_size:] for row in matrix1[:half_size]]
    a21 = [row[:half_size] for row in matrix1[half_size:]]
    a22 = [row[half_size:] for row in matrix1[half_size:]]

    b11 = [row[:half_size] for row in matrix2[:half_size]]
    b12 = [row[half_size:] for row in matrix2[:half_size]]
    b21 = [row[:half_size] for row in matrix2[half_size:]]
    b22 = [row[half_size:] for row in matrix2[half_size:]]

    p1, p1_mult, p1_add = strassen_multiply(a11, subtract_matrices(b12, b22), threshold, level + 1)
    p2, p2_mult, p2_add = strassen_multiply(add_matrices(a11, a12), b22, threshold, level + 1)
    p3, p3_mult, p3_add = strassen_multiply(add_matrices(a21, a22), b11, threshold, level + 1)
    p4, p4_mult, p4_add = strassen_multiply(a22, subtract_matrices(b21, b11), threshold, level + 1)
    p5, p5_mult, p5_add = strassen_multiply(add_matrices(a11, a22), add_matrices(b11, b22), threshold, level + 1)
    p6, p6_mult, p6_add = strassen_multiply(subtract_matrices(a12, a22), add_matrices(b21, b22), threshold, level + 1)
    p7, p7_mult, p7_add = strassen_multiply(subtract_matrices(a11, a21), add_matrices(b11, b12), threshold, level + 1)

    c11 = subtract_matrices(add_matrices(add_matrices(p5, p4), p6), p2)
    c12 = add_matrices(p1, p2)
    c21 = add_matrices(p3, p4)
    c22 = subtract_matrices(subtract_matrices(add_matrices(p1, p5), p3), p7)

    mult_count = p1_mult + p2_mult + p3_mult + p4_mult + p5_mult + p6_mult + p7_mult
    add_count = p1_add + p2_add + p3_add + p4_add + p5_add + p6_add + p7_add
    multiplications += mult_count
    additions += add_count

    result_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            if i < half_size and j < half_size:
                row.append(c11[i][j])
            elif i < half_size and j >= half_size:
                row.append(c12[i][j - half_size])
            elif i >= half_size and j < half_size:
                row.append(c21[i - half_size][j])
            else:
                row.append(c22[i - half_size][j - half_size])
        result_matrix.append(row)

    return result_matrix, multiplications, additions


if __name__ == "__main__":
    #Load
    matrix1 = load_matrix_from_file("matrix1(512).txt")
    matrix2 = load_matrix_from_file("matrix2(512).txt")

    threshold = 2

    start_time = time.time()

    result_matrix_strassen, num_multiplications, num_additions = strassen_multiply(matrix1, matrix2, threshold)

    end_time = time.time()
    run_time = end_time - start_time

    print("\nNumber of multiplications:", num_multiplications)
    print("Number of additions:", num_additions)
    print("\nRun_time:", run_time, "seconds")

    #File
    save_matrix_to_file(result_matrix_strassen, "strassen_multiplication_result(512).txt")