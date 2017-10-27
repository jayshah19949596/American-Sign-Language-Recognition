"""
data operations

This module does data operations like creating and loading of data

"""
import scipy.io as sio
import numpy as np

# ===========================================
# Define all the key variables for dictionary
# ===========================================
full_path = "C:\\Users\\jaysh\\PycharmProjects\\ASL\\Data"
data_str = "data"
hand_str = "handface"
i_d = "id"
lexicon = "lexicon"
_type = "type"
start_frame = "start_frame"
end_frame = "end_frame"
flag = "flag"
signer_id = "signer_id"
group = "group"
directory = "directory"
file_name = "filename"
dominant = "Ld"
non_dominant = "Lnd"
face = "face"


def load_mat_file(file):
	"""
	Load MATLAB file
	
	Parameters
	----------
	file : str
		   Name of the mat file
	       
	Returns
    -------
    mat_dict : dict
       		   dictionary with variable names as keys, and loaded matrices as
    		   values
	"""
	return sio.loadmat(file)


def create_annotation_data(data):
	"""
	Create Annotation data dictionary

	Parameters
	----------
	data : dict
       dictionary with variable names as keys, and loaded matrices as
       values

	Returns
	-------
	data_dic : dict
	   dictionary with properties as keys, and loaded matrices as
	   values
	"""
	data_dic = dict({})
	data_dic[i_d] = data[data_str][0][0][0]
	data_dic[lexicon] = data[data_str][0][0][1]
	data_dic[_type] = data[data_str][0][0][2]
	data_dic[start_frame] = data[data_str][0][0][3]
	data_dic[end_frame] = data[data_str][0][0][4]
	data_dic[flag] = data[data_str][0][0][5]
	data_dic[signer_id] = data[data_str][0][0][6]
	data_dic[group] = data[data_str][0][0][7]
	data_dic[directory] = data[data_str][0][0][8]
	data_dic[file_name] = data[data_str][0][0][9]
	return data_dic


def create_hand_face_data(data):
	"""
	Create hand face data dictionary

	Parameters
	----------
	data : dict
	       dictionary with variable names as keys, and loaded matrices as
	       values

	Returns
	-------
	data_dic : dict
		      dictionary with properties as keys, and loaded matrices as
		      values
	"""
	data_dic = dict({})
	data_dic[hand_str] = data[hand_str]
	return data_dic


def create_data(file):
	"""
		Create hand face data dictionary

		Parameters
		----------
		file : string
	       absolute path to the file that is to be loaded

		Returns
		-------
		data_dic : dict
		   dictionary with properties as keys, and loaded matrices as
		   values
		"""
	data_dic = None
	if "annotation" in file:
		data_dic = create_annotation_data(load_mat_file(file))
	elif "hand" in file:
		data_dic = create_hand_face_data(load_mat_file(file))
	return data_dic


def concatenate_two_hand_data_sets(data_1, data_2):
	for key in data_1:
		print(key)
		print(data_1[key].shape)
		print(data_2[key].shape)
		print(np.vstack((data_1[key], data_2[key])).shape)
	
	
def concatenate_two_annotation_data_sets(data_1, data_2):
	for key in data_1:
		print(key)
		print(data_1[key].shape)
		print(data_2[key].shape)
		print(np.vstack((data_1[key], data_2[key])).shape)
		

if __name__ == '__main__':
	train_notation_data = create_data(full_path+"\\annotation_gb1113.mat")
	train_hand_face_data = create_data(full_path+"\\handface_manual_gb1113.mat")
	test_notation_data = create_data(full_path+"\\annotation_lb1113.mat")
	test_hand_face_data = create_data(full_path+"\\handface_manual_lb1113.mat")
	# concatenate_two_hand_data_sets(train_hand_face_data, test_hand_face_data)
	# concatenate_two_annotation_data_sets(train_notation_data, test_notation_data)
