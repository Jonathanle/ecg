"""
Refactored commends: 

*q1: should I consider adding the dimension or voltage or not? there is must 1 valule? added for consisteny of description

"""

import torch
from torch.utils.data import Dataset

import pandas as pd
from pathlib import Path
import numpy as np

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

	Params: 
	input_dir (str) - directory containing all patient xlsx files
	get_post (str) - retrieve the post xor the pre files
	returns: 
		np.array - np array of shape (num_patients, lead, time_sequence, voltage) *q1 
		filenames - string representation of the filenames
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
	Function that processes the patient labels, returning a dictionary mapping of the patients IDs to the outcome


	Params: 
		input_dir (str): directory to the cohort xlsx file


	Returns (dict): Dictionary containing key mappings to the BinaryResponse
	Variable
	"""
	
	df = pd.read_excel("./data/PatientCohort_ECG.xlsx", header=0, index_col=0)

	
	dict_mapping = df['Binary Response'].to_dict() 

	return dict_mapping


def get_patient_id(filename): 
	"""
	Function to strip the filename and get the Patient ID, this function looks
	for the first instance of an underscore and returns the representation of it
	later on.
	
	Params: 
		filename (str): filename of the format RXX_ ..... .xlsx
	
	Returns: 
		patient id (str) - Patient id with the "RXX" 
	"""
	
	for i in range(0, len(filename)): 
		if filename[i] == "_": 
			return filename[:i]
	raise Exception("Error: no _ foudn in filename")
	
class ECGDataset(Dataset):
    def __init__(self, data_tensor, labels):
        """
        Args:
            data_tensor (torch.Tensor): Tensor containing the data
            labels (list or torch.Tensor): Labels for the data
        """
        self.data = data_tensor

        self.labels = torch.tensor(labels) if not torch.is_tensor(labels) else labels
        
        # Verify data and labels have matching first dimension
        assert self.data.shape[0] == len(self.labels), \
            f"Data size {self.data.shape[0]} doesn't match labels size {len(self.labels)}"

    def __len__(self):
        """Return the total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample
        Returns:
            tuple: (data_sample, label)
        """
        return self.data[idx], self.labels[idx]
def main():
	# preprocess data --> Tensor  TODO: 
	ecg_tensors, filenames = preprocess_directory()
	# filenames are weird --> remove to get only the patient information
	labels = preprocess_patient_labels() # change to a dictionary

	breakpoint()	


	# added this in the main because of a dependency of adding filename to the tests (change in interface leads to bad tests)
	for i in range(len(labels)): # warning: one of the labels is a NON patient id need to account for this TODO 
		labels[i] = get_patient_id(labels[i])

	# TODO: function to remove the information to put into the dataset	

	# TODO: Combine the processes here into a Pytorch Dataset




	# traansforms the data into an interface.



	# define model
	# model train on data.  


	return
if __name__ == '__main__':
    main()
