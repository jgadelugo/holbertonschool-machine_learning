#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
plt.hist(student_grades, edgecolor='black', bins=10)
plt.title("Project A")
plt.ylabel('Number of Students')
plt.xlabel('Grades')

plt.show()
