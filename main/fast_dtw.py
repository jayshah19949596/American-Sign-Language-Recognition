import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
y = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance, path)

# import numpy as np
#
# a = [[1, 2], [3, 4], [5, 6], [6, 8]]
# a = np.array(a)
#
# print(a)
# print()
#
# square_pts = np.square(a)
# print()
# print(square_pts)
#
# sum_of_pts = np.sum(a, axis = 1)
# sum_of_pts = np.transpose(np.asmatrix(sum_of_pts))
# print()
# print(sum_of_pts)
#
# square_root = np.sqrt(sum_of_pts)
# print()
# print(square_root)
#
# unit= a/square_root
# print()
# print(unit)
