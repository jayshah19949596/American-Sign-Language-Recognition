import numpy as np


def find_centroid(data):
	"""
	Finds the centroid of bounding boxes

	Parameters
	----------
	data : 2-D numpy array
		   Numpy array with bounding box data of format [y, x, width, height]
		   where (x, y) is top left corner of the bounding box

	Returns
	-------
	centroid : 2-D numpy array
			   Numpy array with centroid of format [x, y]
	"""
	# [y, x, width, height]
	centroid_x = (data[:, 0] + data[:, 3])/2  # (x_coordinate + height)/2
	centroid_y = (data[:, 1] + data[:, 2])/2  # (y_coordinate + width) /2
	centroid = np.transpose(np.vstack((centroid_x, centroid_y)))  # [[x1, y1], [x2, y2],...., [xn, yn]]
	return centroid


def find_relative_coordinates(new_origin, coordinates):
	return new_origin - coordinates


def get_face_diagonal(x, y):
	return np.sqrt(x*x + y*y)
