"""
Refactored commends: 

*q1: should I consider adding the dimension or voltage or not? there is must 1 valule? added for consisteny of description

"""

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim import Adam
# import torch.nn.functional as F

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pandas as pd
from pathlib import Path

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
	

		ecg_training_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
		label = torch.tensor(self.labels[patient_id], dtype=torch.int64).unsqueeze(0) # need to have an unsqueezed for shape documentation


		return ecg_training_tensor, label, patient_id 

class ECGNet(nn.Module): # TODO: Complete but do not test + learn and understand the code later
    """
    Network that performs 1d convolutions to combine informaation locally to transform information to another representation that is useful
    """
    def __init__(self, num_leads=12, seq_length=250):

	
        super(ECGNet, self).__init__()
        
        # Parameters
        self.num_leads = num_leads
        self.seq_length = seq_length
        
        # Initial per-lead feature extraction
        self.lead_features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)
        )
        
        # Calculate the size after convolutions and pooling
        self.feature_length = seq_length // 8  # After 3 max pooling layers
        
        # Combine features across leads
        self.combine_leads = nn.Sequential(
            nn.Linear(128 * self.feature_length * num_leads, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Process each lead independently
        # Reshape to process leads independently: (batch, leads, time, 1) -> (batch * leads, 1, time)
        x = x.permute(0, 1, 3, 2).reshape(-1, 1, self.seq_length)
        
        # Extract features from each lead
        x = self.lead_features(x)
        
        # Reshape to combine lead features
        x = x.reshape(batch_size, self.num_leads, -1)
        x = x.reshape(batch_size, -1)  # Flatten for fully connected layers
        
        # Combine features from all leads
        x = self.combine_leads(x)
        x = self.sigmoid(x)
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    for data, labels, _ in train_loader:
        data, labels = data.to(device), labels.to(device)
        labels = labels.float().to(device)  # Convert to float for BCE loss note for later + add to device
        
        optimizer.zero_grad()
        outputs = model(data) # TODO: verify with testing that the outputs are inside the gpu not cpu (need a bottleneck flow showing the data *in* specific spots
		

        loss = criterion(outputs, labels) # TODO: analyze and formalize behavior of test in this in the test what exactly need the inputs to be? )
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(true_labels, predictions)
    
    return epoch_loss, epoch_auc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, labels, _ in val_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.float()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_auc = roc_auc_score(true_labels, predictions)
    
    return val_loss, val_auc

def train_model(dataset, model, num_epochs=100, batch_size=16, learning_rate=0.001):
    """
    Robust training loop with input validation and error checking
    
    Args:
        dataset (ECGDataset): The dataset
        model (nn.Module): The model to train
        num_epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        
    Returns:
        dict: Training history and best model state
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate inputs
    if not isinstance(dataset, ECGDataset):
        raise TypeError("Dataset must be instance of ECGDataset")
    if batch_size > len(dataset):
        raise ValueError(f"Batch size ({batch_size}) larger than dataset size ({len(dataset)})")
        
    model = model.to(device)
    
    # Create data splits with validation
    splits = create_data_splits(dataset)
    min_split_size = min(len(splits['train']), len(splits['val']))
    if min_split_size < batch_size:
        raise ValueError(f"Split size ({min_split_size}) smaller than batch size ({batch_size})")
    
    # Create data loaders with error checking
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(splits['train']),
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=False  # Keep all samples
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(splits['val']),
        num_workers=0,
        drop_last=False
    )
    
    # Initialize training components
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training state
    best_val_auc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': []
    }
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation phase
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # Model selection
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break
                    
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        if best_model_state is not None:
            print("Recovering best model state...")
            model.load_state_dict(best_model_state)
    
    # Always restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'history': history,
        'best_val_auc': best_val_auc
    }



def create_data_splits(dataset, n_splits=5, fold_idx=0, val_size=0.2):
    """
    Create train/val/test splits that can be extended to k-fold CV.
    Initially uses only one fold but structured for easy k-fold extension.
    
    Args:
        dataset: ECGDataset instance
        n_splits: Number of folds (default=5)
        fold_idx: Which fold to use as test set (default=0)
        val_size: Size of validation set as fraction of training data
    
    Returns:
        dict containing train, val, and test indices
    """
    # Get labels for stratification
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label.item())
    labels = np.array(labels)
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get indices for the specified fold
    splits = list(skf.split(np.zeros(len(dataset)), labels))
    train_idx, test_idx = splits[fold_idx]
    
    # Further split training data into train and validation
    val_size = int(len(train_idx) * val_size)
    val_idx = train_idx[:val_size]
    train_idx = train_idx[val_size:]
    
    return {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }


def main():
	ecg_tensors, filepaths = preprocess_directory()
	
	training_data_file_ids = get_patient_ids(filepaths)# helper fucntion due to dependency
	
	# filenames are weird --> remove to get only the patient information
	labels = preprocess_patient_labels() # change to a dictionary

	
	dataset = ECGDataset(ecg_tensors, training_data_file_ids, labels)
	
	splits = create_data_splits(dataset, n_splits=5, fold_idx=0) # strategy - using pointers for spllits is much more useful than actually splitting
	train_loader = DataLoader( dataset, 
		batch_size=32, 
		sampler=SubsetRandomSampler(splits['train']), # notice here that the train variable greatly simplifies the approach (pseudo discovery based)
		num_workers=0
	)
	val_loader = DataLoader(
		dataset, 
		batch_size=32,
		sampler=SubsetRandomSampler(splits['val']),
		num_workers=0
	)
	test_loader = DataLoader(
		dataset, 
		batch_size=32,
		sampler=SubsetRandomSampler(splits['test']),
		num_workers=0
	)



	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	assert device == torch.device('cuda') # assuemd device to be a string when it was specificallly torch.device('cuda') wrapper


	model = ECGNet().to(device)

	class_weights = torch.Tensor([0.8]).to(device)# TODO: replace this with class weightsdataset.get_class_weights() error the weights here influenced something about BCE (entroyp of the mind)
	# TODO: Transform Uncertainty of my representation of how tensor works to --> certainty about the workings
	# class weights - VERY IMPORTANT to consider the class weights and where they are, the hardware interface is shown here. 
	criterion = torch.nn.BCELoss(weight=class_weights)  # subjective uncertaainties - what is BCE data type inputs? class weights why impt? 
	# df is BCE loss required to have a label output? whawt is teh output
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# 6. Training loop
	num_epochs = 100
	best_val_auc = 0
	best_model_state = None
	patience = 10
	patience_counter = 0

	for epoch in range(num_epochs):
		# Train
		train_loss, train_auc = train_epoch( 
		    model=model,
		    train_loader=train_loader,
		    criterion=criterion,
		    optimizer=optimizer,
		    device=device
		)

		# Validate
		val_loss, val_auc = validate(
		    model=model,
		    val_loader=val_loader,
		    criterion=criterion,
		    device=device
		)

		# Print metrics
		print(f'Epoch {epoch+1}/{num_epochs}:')
		print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
		print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

		# Model selection and early stopping
		if val_auc > best_val_auc:
			best_val_auc = val_auc
			best_model_state = model.state_dict().copy()
			patience_counter = 0

			# Save best model
			save_dir = Path('models')
			save_dir.mkdir(exist_ok=True)
			torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'val_auc': val_auc,
			}, save_dir / 'best_model.pt')
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print('Early stopping triggered')
				break




	return
if __name__ == '__main__':
    main()
