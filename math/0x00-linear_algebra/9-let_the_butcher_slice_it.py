#!/usr/bin/env python3
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 = matrix[1:3]
mat2 = np.array([x[2:4] for x in matrix])
mat3 = np.array([x[-3:] for x in matrix[1:]])
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))