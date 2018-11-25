import numpy as np

x = np.array([[1, 2, 3], [1, 2, 3]])  # 2x3

a = np.eye(3)

temp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3

print(x.dot(a).dot(temp))

print(temp.T.dot(a).dot(x.T))


print(x.dot(temp).T)

print(temp.T.dot(x.T))
