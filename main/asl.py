import ops.data_operations as dop
from my_dtw import DTW

if __name__ == "__main__":
	
	train_notation_dat = dop.create_data(dop.full_path + "\\annotation_gb1113.mat")
	train_hand_face_data = dop.create_data(dop.full_path + "\\handface_manual_gb1113.mat")
	test_notation_data = dop.create_data(dop.full_path + "\\annotation_lb1113.mat")
	test_hand_face_data = dop.create_data(dop.full_path + "\\handface_manual_lb1113.mat")
	
	dtw = DTW(train_hand_face_data[dop.hand_str], test_hand_face_data[dop.hand_str])
	dtw.apply_dtw()
