import numpy as np

x = np.array([[1, 2, 3]])

temp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

y = np.array([[4, 5, 6]])

print(np.dot(y, temp).dot(x.T))

print(np.dot(temp.T, y.T).dot(x))

