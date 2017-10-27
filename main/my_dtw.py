import ops.data_operations as dop
from preprocess import pre_processing as pp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time
from dtw import dtw as _dtw
import numpy as np
from tqdm import tqdm

# TRAIN NOTATION = annotation_gb1113.mat
# TRAIN HAND     = handface_manual_gb1113.mat
# TEST NOTATION  = annotation_lb1113.mat
# TEST HAND      = handface_manual_lb1113.mat

# The code will run approximately for 6.18333 hours
# one test data takes around 20 seconds

# ===================================
# First time it took 4.625 hours
# Got an accuracy of 0.3926325247079964 with only Dominant hands
# Got 437 videos correct out of 1113
# These results are with DTW

# Sixth time it took 5.3 hours
# Got an accuracy of 0.39802336028751123 with Dominant hands
# Got 443 videos correct out of 1113
# These results are with fast_dtw
# ===================================

# Second time it took 7.6 hours
# Got an accuracy of 0.48247978436657685 with Dominant hands and Non-Dominant Hands
# Got 534 videos correct out of 1113
# These results are with DTW

# Fifth time it took 4.34 hours
# Got an accuracy of 0.4950584007187781 with Dominant hands, Non-Dominant Hands, and LDelta
# Got 599 videos correct out of 1113
# These results are with fast_dtw
# ===================================

# Third time it took 10.8 hours
# Got an accuracy of 0.5381850853548967 with Dominant hands, Non-Dominant Hands, and LDelta
# Got 599 videos correct out of 1113
# These results are with DTW

# Fourth time it took 6.12 hours
# Got an accuracy of 0.5381850853548967 with Dominant hands, Non-Dominant Hands, and LDelta
# Got 599 videos correct out of 1113
# These results are with fast_dtw
# ===================================

# Seventh time it took 32.27 hours
# Got an accuracy of 0.2884097035040431 with 3 features & Face Normalization
# Got 321 videos correct out of 1113
# These results are with DTW

# ============================
# Eight time it took 11.09 hours
# Got an accuracy of 0.5579514824797843 with 3 features
# Got 621 videos correct out of 1113
# These results are with DTW

# ============================
# Ninth time it took 11.09 hours
# Got an accuracy of 0.5669362084456424 with [ld, lnd, l_delta, od, ond, correct co-ordinates]
# Got 631 videos correct out of 1113
# These results are with DTW


class DTW(object):
	def __init__(self, train_hand_face, test_hand_face, train_notation, test_notation):
		self.train_hand_face_data = train_hand_face
		self.test_hand_face_data = test_hand_face
		self.train_notation_data = train_notation
		self.test_notation_data = test_notation
		self.top_k = 10
		self.correct = 0
		
	def apply_dtw(self):
		"""
		Computes Dynamic Time Warping (DTW) of all the testing data with training data
		and displays the accuracy
		"""
		start = time.clock()
		# train_data = test_data.shape = (1113, 3) i.e.(no_of_video, hand_face_data)
		train_data, test_data = self.train_hand_face_data, self.test_hand_face_data
		train_notation, test_notation = self.train_notation_data, self.test_notation_data
		train_size = train_data.shape[0]
		test_size = test_data.shape[0]
		
		for i in tqdm(range(test_size)):
			# i = 25
			distances = []
			test_ld, test_lnd, test_face, test_lnd_is_empty, test_od, test_ond = perform_pre_processing(test_data, i)
			test_target = test_notation[dop.lexicon][i][0][0]
			# print(test_target, test_lnd_is_empty)

			for j in range(train_size):
				total_distance = 0
				train_ld, train_lnd, train_face, train_lnd_is_empty, train_od, train_ond = perform_pre_processing(train_data, j)
				if train_lnd_is_empty == test_lnd_is_empty:
					# print(train_ld.shape, test_ld.shape)
					if train_lnd_is_empty:
						total_distance += get_dtw_distance(train_ld, test_ld)
						total_distance += get_dtw_distance(train_od, test_od)

					elif not train_lnd_is_empty:
						total_distance += get_dtw_distance(train_ld, test_ld)
						total_distance += get_dtw_distance(train_lnd, test_lnd)
						total_distance += get_dtw_distance(train_od, test_od)
						total_distance += get_dtw_distance(train_ond, test_ond)
						total_distance += get_dtw_distance((train_ld-train_lnd), (test_ld-test_lnd))
						
					distances.append([total_distance, train_lnd_is_empty, train_notation[dop.lexicon][j][0][0]])
				
				else:
					continue
				# break
			self.evaluate_prediction(test_target, distances)
			break
		
		self.display_accuracy(test_size)
		end = time.clock()
		print(end - start)
		
	def display_accuracy(self, test_size):
		"""
		Displays the final accuracy
		"""
		print("Accuracy :", (self.correct/test_size))
		
	def evaluate_prediction(self, test_target, distances):
		"""
		counts how many times did DTW gave the correct prediction in top k results
		
		Parameters
		----------
		:param test_target: numpy array
	    	    	        having a single element which is of type string
	    	    	        it indicates the type of sign of the test data
	       	
	    :param distances: list
	    	   			  having a distances of all the training data with a particular test data
	    	   			  	       	    
		"""
		distances = sorted(distances, key=lambda x: x[0])
		distances = distances[0: self.top_k]
		distances = np.array(distances)
		if test_target in distances[:, 2]:
			self.correct += 1
			
			
def get_dtw_distance(x, y):
	dist, cost, acc, path = _dtw(x, y, dist=euclidean)
	return dist


def get_fast_dtw_distance(x, y):
	dist, path = fastdtw(x, y, dist=euclidean)
	return dist
	

def perform_pre_processing(data, i):
	ld = data[i][0].astype('float64')
	ld = pp.find_centroid(ld)
	ond = []
	face = data[i][2][0].astype('float64')
	
	face = face.reshape(1, face.shape[0])
	face = pp.find_centroid(face)
	
	lnd = data[i][1].astype('float64')
	if lnd.size == 0:
		lnd_is_empty = True
	else:
		lnd_is_empty = False
		lnd = pp.find_centroid(lnd)
	
	ld = pp.find_relative_coordinates(face, ld)
	
	od = get_vec_direction_of_motion(ld)

	if not lnd_is_empty:
		lnd = pp.find_relative_coordinates(face, lnd)
		ond = get_vec_direction_of_motion(lnd)
		
	return ld, lnd, face, lnd_is_empty, od, ond


def get_vec_direction_of_motion(data):
	# print(data)
	current_frames = data[0: data.shape[0]-1, :].astype('float64')
	previous_frames = data[1:, :].astype('float64')
	final_frames = (previous_frames - current_frames).astype('float64')
	square_of_points = np.square(final_frames)
	sum_of_points = np.sum(square_of_points, axis=1)
	sum_of_points = np.transpose(np.asmatrix(sum_of_points))
	square_root = np.sqrt(sum_of_points)
	# print(square_root)
	# print("=====================")
	# print(final_frames)
	square_root[square_root == 0.0] = 0.0001
	final_frames = final_frames/square_root
	# print("=====================")
	# print(final_frames)
	return final_frames


def data_processing(data):
	print(data[dop.hand_str].shape)
	
if __name__ == '__main__':
	train_notation_data = dop.create_data(dop.full_path+"\\annotation_gb1113.mat")
	train_hand_face_data = dop.create_data(dop.full_path+"\\handface_manual_gb1113.mat")
	test_notation_data = dop.create_data(dop.full_path+"\\annotation_lb1113.mat")
	test_hand_face_data = dop.create_data(dop.full_path+"\\handface_manual_lb1113.mat")
	# print(train_notation_data[dop.lexicon][0][0][0])
	# print(train_notation_data[dop.lexicon][1][0][0])
	# print(train_notation_data[dop.lexicon][2][0][0])
	dtw = DTW(train_hand_face_data[dop.hand_str], test_hand_face_data[dop.hand_str],
			  train_notation_data, test_notation_data)
	dtw.apply_dtw()
