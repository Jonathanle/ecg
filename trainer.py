"""
Refactored commends: 

*q1: should I consider adding the dimension or voltage or not? there is must 1 valule? added for consisteny of description

"""

import torch
from torch.utils.data import Dataset

import pandas as pd
from pathlib import Path
import numpy as np

import re

def preprocess_data_file(filepath):
    """
    Process 1 single xlsx file to retrieve the pd dataframe

    Params: 
        filepath - path to xlsx file for interpreting


    returns: 
        pandas df object (TODO Modify thisi file to change to a numpy matrix)

    Directory Format: 
        
    dir - 1 xlsx file
    
    Patient File Requirements:
        MUST have 12 columns corresponding to leads 0-12
        MUST have the standardized 250 time samples


    this will process 1 patient file only* further preprocessing
    of multiple will be done later
    """

    NUM_TIME_STEPS = 250 
    NUM_LEADS = 12

    df = pd.read_excel(filepath, header = None, index_col = None)

    if df.shape == (251, 12): 
        breakpoint()	

    assert df.shape == (NUM_TIME_STEPS, NUM_LEADS), f"error: num_timesteps = {df.shape[0]} num_leads = {df.shape[1]}"


    return df


def preprocess_directory(input_dir="./data/InterpolatedQRS/", get_post=False):
	"""
	Processes all the xlsx files and returns a group of the pytorch
	datasets

	Processes files with satisfying the regex "Rd+" so only anything with R20 R100 etc.

	Params: 
	input_dir (str) - directory containing all patient xlsx files
	get_post (str) - retrieve the post xor the pre files
	returns: 
		np.array - np array of shape (num_patients, lead, time_sequence, voltage) *q1 
		filenames - string representation of the filenames filtered or not?
			note these - some tests rely on it being filenames
			
	"""

	# Convert input directory to Path object for easier handling
	input_path = Path(input_dir)

	# Get all xlsx files in the directory
	# Filter based on whether we want post or pre files
	file_pattern = "*post*.xlsx" if get_post else "*pre*.xlsx"
	patient_files = list(input_path.glob(file_pattern))

	# Process each file and collect the numpy arrays
	patient_arrays = []
	file_path_arrays = []
	for posix_file_path in patient_files:
		# Assuming you have a function like process_single_file that returns
		# an array of shape (lead, time_sequence, voltage)
		patient_data = preprocess_data_file(posix_file_path)  

		if len(get_patient_id(str(posix_file_path))) == 0: 
			continue # Filter Clin Trials

		patient_arrays.append(patient_data)
		file_path_arrays.append(str(posix_file_path)) # changed this to get the str because file_path is a posix path NOT an str

		

	# Stack all patient arrays along the first dimension
	# This will create an array of shape (num_patients, lead, time_sequence, voltage)
	if patient_arrays:
		combined_array = np.stack(patient_arrays, axis=0)
		combined_array = np.expand_dims(combined_array, axis=-1)
	else:
		raise ValueError(f"No {'post' if get_post else 'pre'} files found in {input_dir}") 

	return combined_array, file_path_arrays

def preprocess_patient_labels(input_dir="./data/PatientCohort_ECG.xlsx"): 
	"""
	Function that processes the patient labels, returning a dictionary mapping of the patients IDs to the 'Binary Response' column
	Patient IDs that are not of the form "RX" or "RXX" or "RXX" are discarded if R?? has non digits it is also rejected
		
	If the corresponding patient values not 0 or 1 reject

	Params: 
		input_dir (str): directory to the cohort xlsx file


	Returns: 
		dict[str, int],  Dictionary containing key mappings to the BinaryResponse
	Variable
	""" 
	VALUE_COLUMN= 'Binary Response'

	df = pd.read_excel("./data/PatientCohort_ECG.xlsx", header=0, index_col=0)
	
	dict_mapping = df[VALUE_COLUMN].to_dict() 

	new_dict = {}	
	for key, value in dict_mapping.items():

		condition = isinstance(key, str) and len(key) >= 2 
		if (isinstance(key, str) and
			len(key) >= 2 and 
			key[0] == "R" and 
			key[1].isdigit() and
			(value == 0 or value == 1)): 
			
			new_dict[key] = value			



	return new_dict


def get_patient_id(filepath): 
	"""
	Function to strip the filename and get the Patient ID, this function looks
	for the first instance of an underscore and returns the representation of it
	later on.
	
	Params: 
		filename (str): filename of the format RXX_ ..... .xlsx
	
	Returns: 
		str - Patient id with the "RXX" 
	"""
	pattern = r'R\d+'
	
	matches = re.findall(pattern, filepath)
	
	return matches

	

def get_patient_ids(filepaths): 
	"""
	Function to retrieve all the patient ids from a python list 

	Params: 
		filepaths (list) - python list of strings corresponding to filenames 
	Returns:
		new_patient_ids (list) - python list containing patient ids
	"""
	
		
	new_patient_ids = []

	for filepath in filepaths:
		patient_id_list = get_patient_id(filepath)	
		if len(patient_id_list) != 1: 
			breakpoint()
		assert len(patient_id_list) == 1, "error: len patient_id list ne 1" # one hit (extremely valuable caught the fact that Clintrial was also a found file assumed before that it was only RXX files found

		new_patient_ids.append(patient_id_list[0])
	
	
	return new_patient_ids



class ECGDataset(Dataset):
	def __init__(self, data_tensor, data_patient_ids, labels):
		"""
		Args:
		    data_tensor (torch.Tensor): Tensor containing the data
		    data_patient_ids (list): list with ids corresponding to data tensor indices
		    labels (dict[patient_id, binary_outcome): list for data

		This dataset will correct and match training data only to the labels 
		reconstructing the dataset for now it will be unorganized and unrefactored
		DF 
		"""
		
		self.data, self.patient_id_idx_mapping, self.labels = self.create_data_interface(data_tensor, data_patient_ids, labels)
	

		self.idx_patient_mapping = {idx: patient_id for (patient_id, idx) in self.patient_id_idx_mapping.items()}

		# Verify data and labels have matching first dimension
		assert self.data.shape[0] == len(self.labels), \
		    f"Data size {self.data.shape[0]} doesn't match labels size {len(self.labels)}"


	def create_data_interface(self, data, data_patient_ids, labels): 
		"""
		Create pytorch datasets # TODO TEST + better document precisely interactions + interface assume harder to refactor higher cost
		to fix from entropy + need to meet urgent deadlines

		Assumes alll data for labels exists in training data	

		Params: 
		    labels
		    data


		feelings - it just feells loose and thata "something" might happen but it still gets the main job done.
		"""

		patient_training_index_mapping = {}
		
		# entropy from the data structure transformations df
		num_data = len(labels.items())
		new_tensor_shape = list(data.shape)
		new_tensor_shape[0] = num_data

		new_tensor = np.zeros(new_tensor_shape)


		# create patient data mapping
		training_data_patient_mapping = {}
		for i in range(len(data_patient_ids)):

			patient_id = data_patient_ids[i] 
			training_data_patient_mapping[patient_id] = i 

		for i, (patient_id, label) in enumerate(labels.items()): 
			data_index = training_data_patient_mapping[patient_id]

			data_tensor = data[data_index]

			new_tensor[i] = data_tensor
			patient_training_index_mapping[patient_id] = i

		return new_tensor, patient_training_index_mapping, labels


	def __len__(self):
		"""Return the total number of samples"""
		return len(self.data)

	def __getitem__(self, idx):
		"""
		Args:
		    idx (int): Index of the data sample
		Returns:
		    tuple: (data_sample, label, patient_id str)
		"""
		
		patient_id = self.idx_patient_mapping[idx]
		return self.data[idx], self.labels[patient_id], patient_id
def main():
	ecg_tensors, filepaths = preprocess_directory()
	
	training_data_file_ids = get_patient_ids(filepaths)# helper fucntion due to dependency
	
	# filenames are weird --> remove to get only the patient information
	labels = preprocess_patient_labels() # change to a dictionary

	
	# TODO: function to remove the information to put into the dataset	
	# TODO: Combine the processes here into a Pytorch Dataset
	dataset = ECGDataset(ecg_tensors, training_data_file_ids, labels)
	
	breakpoint()	
	# prototype manual system testing other unknown errors can manifest later



	# traansforms the data into an interface.



	# define model
	# model train on data.  


	return
if __name__ == '__main__':
    main()
