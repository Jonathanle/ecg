"""
Refactored commends: 

*q1: should I consider adding the dimension or voltage or not? there is must 1 valule? added for consisteny of description

"""

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torch.nn.functional as F


from torch.optim import Adam
# import torch.nn.functional as F


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np



import pandas as pd
from pathlib import Path
import re
import inspect # function used for checking code structure at runtime

import matplotlib.pyplot as plt

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



def preprocess_disease_labels(input_dir, class1=None, class2=None, header=0):
    """
    Function that processes the disease labels, returning a dictionary mapping patient IDs
    to binary labels for a classification task between two specified disease classes.

    Only includes patients that have a positive label for either class1 or class2 (or both).
    Patients with 0 in both class1 and class2 columns are excluded.
        
    This fucntion knows specifically that id columsn are integers, not with standard R 

    Params:
        input_dir (str): Path to the Excel file containing disease data
        class1 (str): First disease class column name ('GEN', 'MYO', or 'SARC')
        class2 (str): Second disease class column name ('GEN', 'MYO', or 'SARC')
        header (int): Row index to use as column headers (default 0)
        
    Returns:
        dict[str, int]: Dictionary with patient IDs as keys and binary labels as values
                        1 if patient has class1, 0 if patient has class2
    """
    # Validate class inputs
    valid_classes = ['GEN', 'MYO', 'SARC']
    if class1 not in valid_classes or class2 not in valid_classes:
        raise ValueError(f"Both class1 and class2 must be one of {valid_classes}")
    if class1 == class2:
        raise ValueError("class1 and class2 must be different")

    # Read the Excel file
    df = pd.read_excel(input_dir, header=header)

    # Create an empty dictionary for the mapping
    label_dict = {}

    # Process each row in the dataframe
    for _, row in df.iterrows():
        patient_id = str(row['ID'])
        patient_id = "R" + patient_id
        
        # Check if patient has either class1 or class2
        has_class1 = row[class1] == 1
        has_class2 = row[class2] == 1
        
        # Skip patients that don't have either disease
        if not has_class1 and not has_class2:
            continue
            
        # For patients with class1, assign label 1
        # For patients with only class2, assign label 0
        # If a patient has both, prioritize class1
        if has_class1:
            label_dict[patient_id] = 1
        else:
            label_dict[patient_id] = 0
            
    return label_dict
def preprocess_patient_labels(input_dir): # removed the input_dir because I thought that here, i remove requirements of any knowledge that hides bugs, this must* always have a dependency on the input it inputs 
    
    labels = preprocess_disease_labels(input_dir, class1="GEN", class2="MYO")

    return labels
def preprocess_patient_labels_old(input_dir="./data/PatientCohort_ECG.xlsx"): 
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

    df = pd.read_excel(input_dir, header=0, index_col=0)
    
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


class ECGCompositeDataset(Dataset):

    def __init__(self, pre_dataset, post_dataset):
        self.pre_dataset = pre_dataset
        self.post_dataset = post_dataset


        assert len(pre_dataset) == len(post_dataset), "Error: len of predataset not equal to post_dataset"
        

    def __len__(self): 
        return len(self.pre_dataset)
    def __getitem__(self, idx): 
        """
        Implement a O(n) search with respect to dataset - (wont matter because the dataset size is fixed anyways)
        """
        pre_tensor, labels, patient_id_pre = self.pre_dataset[idx]
        
        for i in range(0, len(self.post_dataset)):
            post_tensor, _, patient_id_post = self.post_dataset[i]
            if patient_id_post == patient_id_pre: 
                return torch.stack((pre_tensor, post_tensor), axis=0), labels, patient_id_post  
        
        raise Exception(f"Error: for patient id {patient_id_pre} no id found in post dataset")


class ECGNet(nn.Module): # TODO: Complete but do not test + learn and understand the code later (reduce entropy by learaning how the functions work
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
    def encode_leads(self, x): 

        batch_size = x.shape[0]
        
        # Process each lead independently
        # Reshape to process leads independently: (batch, leads, time, 1) -> (batch * leads, 1, time)
        x = x.permute(0, 1, 3, 2).reshape(-1, 1, self.seq_length) # WTF did i do here?
        
        # Extract features from each lead
        x = self.lead_features(x) # Hard to represent these idea formallly how would i define something of "good representation" as good? 
        
        # Reshape to combine lead features
        x = x.reshape(batch_size, self.num_leads, -1) # TODO: having better models to then better approximate and explain known empirical experiences -> creating computational graphs
        x = x.reshape(batch_size, -1)  # Flatten for fully connected layers
        return x

    def forward(self, x):
        x = self.encode_leads(x) 
        # Combine features from all leads
        x = self.combine_leads(x)
        x = self.sigmoid(x)
        
        return x

class ECGCompositeNet(nn.Module):
    def __init__(self, num_leads=12, seq_length=250):

        super(ECGCompositeNet, self).__init__()


        self.ecg_net_pre = ECGNet(num_leads=num_leads, seq_length=seq_length)
        self.ecg_net_post = ECGNet(num_leads=num_leads, seq_length=seq_length)

        # 3/28 changted dropout to employ heaavier regularization in model  
        # 0.5 --> 0.8, 0.3 --> 0.5
        self.combine_leads_pre = nn.Sequential(
            nn.Linear(128 * self.ecg_net_pre.feature_length * self.ecg_net_pre.num_leads, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32)
        )
        
        # 3/28 changed this as well
        self.combine_leads_post = nn.Sequential(
            nn.Linear(128 * self.ecg_net_post.feature_length * self.ecg_net_post.num_leads, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32)
        )
        # TODO: Create computation combining the leads

        self.eval_head = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        # Manually tested TODO if necessary formalize the behavior later if changing
        x_pre_input = x[:, 0, :, :, :].squeeze(1)
        x_post_input = x[:, 1, :, :, :].squeeze(1)
       
        # From the network, use specifically the encoder part of the ECG net NOT the decoder. 
        # TODO: SB see how I can get interpretations for these in the last layer.
        x_pre = self.ecg_net_pre.encode_leads(x_pre_input) # need to feed in batch size assumed the first dimension is the batch size
        x_post = self.ecg_net_post.encode_leads(x_post_input)

        # Summarize the Information global using MLP Encoding Funtion to maintain feature rich representation
        x_pre = self.combine_leads_pre(x_pre)
        x_post = self.combine_leads_post(x_post)

        # Combine with a single linear layer evaluating everything

        x_combined = torch.cat((x_pre, x_post), dim=1)  #torch.stack((x_pre, x_post), axis=0).flatten?  # what am i missing here
        output_logit = self.eval_head(x_combined)
        
        prob_output = self.sigmoid(output_logit)

        return prob_output 

def gradcam_ecg(model, ecg_input, target_class=None, use_post=False):
    """
    Generate GradCAM heatmaps for each lead in the ECG data
    
    Args:
        model: ECGNet or ECGCompositeNet model
        ecg_input: Input ECG tensor 
                  For ECGNet: (batch, leads, time, 1)
                  For ECGCompositeNet: (batch, 2, leads, time, 1)
        target_class: Class to explain (None = use predicted class)
        use_post: For composite model, whether to analyze post ECG (True) or pre ECG (False)
        
    Returns:
        lead_heatmaps: GradCAM heatmaps for each lead (batch, leads, time)
    """
    model.eval()
    
    # Get appropriate network based on model type
    if isinstance(model, ECGCompositeNet):
        if use_post:
            # Extract post ECG input and target network
            ecg_net = model.ecg_net_post
            # For composite model, extract the post ECG data
            if ecg_input.dim() == 5:  # (batch, 2, leads, time, 1)
                ecg_data = ecg_input[:, 1, :, :, :]  # Get post data
            else:
                raise ValueError("Expected 5D input for composite model")
        else:
            # Extract pre ECG input and target network
            ecg_net = model.ecg_net_pre
            # For composite model, extract the pre ECG data
            if ecg_input.dim() == 5:  # (batch, 2, leads, time, 1)
                ecg_data = ecg_input[:, 0, :, :, :]  # Get pre data
            else:
                raise ValueError("Expected 5D input for composite model")
    else:
        # For single ECGNet, use the model directly
        ecg_net = model
        ecg_data = ecg_input
    
    # Ensure input is in the right shape
    if ecg_data.dim() == 4:  # (batch, leads, time, 1)
        pass  # Already in the right format
    else:
        raise ValueError("Expected 4D input for ECG data")
   
    
    # ECG data somehow is 1, 250, 12, 1
    batch_size = ecg_data.shape[0]
    num_leads = ecg_data.shape[2]
    time_steps = ecg_data.shape[1]
    
    # Storage for activations and gradients
    activations = []
    gradients = []
    
    # Find the last convolutional layer in lead_features
    # We know from the model printout it's at index 8
    last_conv_layer = ecg_net.lead_features[8]  # Conv1d(64, 128)
    
    # Forward hook to capture activations
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    # Backward hook to capture gradients
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    backward_handle = last_conv_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    if isinstance(model, ECGCompositeNet):
        output = model(ecg_input)
    else:
        output = model(ecg_data)
    
    # Determine target class if not specified
    if target_class is None:
        target_class = (output > 0.5).float()
    
    # Backward pass to get gradients
    if target_class == 1:
        model.zero_grad()
        output.backward(retain_graph=True)
    else:
        model.zero_grad()
        (1 - output).backward(retain_graph=True)
    
    # Clean up hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Process activations and gradients
    activation = activations[0]  # Shape: (batch*leads, channels, time)
    gradient = gradients[0]      # Shape: (batch*leads, channels, time)
    
    # Reshape to separate leads
    # Original lead reshaping in model: 
    # x = x.permute(0, 1, 3, 2).reshape(-1, 1, self.seq_length)
    # So to reverse, we need to reshape (batch*leads, channels, time) -> (batch, leads, channels, time)

    activation = activation.reshape(batch_size, num_leads, activation.shape[1], activation.shape[2])
    gradient = gradient.reshape(batch_size, num_leads, gradient.shape[1], gradient.shape[2])
    
    # Calculate weights for each feature map via global average pooling
    weights = torch.mean(gradient, dim=3)  # (batch, leads, channels)
    
    # Generate CAM for each lead
    cam = torch.zeros((batch_size, num_leads, activation.shape[3]), device=ecg_data.device)
    
    for b in range(batch_size):
        for l in range(num_leads):
            # Weight each channel's activation by its importance
            weighted_activations = weights[b, l, :, None] * activation[b, l, :, :]  # (channels, time)
            # Sum across channels
            lead_cam = torch.sum(weighted_activations, dim=0)  # (time)
            # Apply ReLU to focus on features that positively influence the decision
            lead_cam = F.relu(lead_cam)
            cam[b, l, :] = lead_cam
    
    # Normalize CAM for each lead
    for b in range(batch_size):
        for l in range(num_leads):
            lead_cam = cam[b, l, :]
            if lead_cam.max() > 0:
                # Min-max normalization to [0,1]
                cam[b, l, :] = (lead_cam - lead_cam.min()) / (lead_cam.max() - lead_cam.min() + 1e-8)
    
    # Upsample to original time dimension
    # cam shape: (1, 12, 62)
    # Interpolate / Guess in Data Gaps to have a 250 time sequence
    cam = F.interpolate(
        cam,
        size=time_steps,
        mode='linear',
        align_corners=False
    ).squeeze(1)
    
    return cam


def visualize_lead_gradcam(ecg_data, gradcam_maps, patient_id=None, lead_names=None):
    """
    Visualize ECG with GradCAM heatmap overlays for each lead
    
    Args:
        ecg_data: ECG data tensor of shape (leads, time, 1)
        gradcam_maps: GradCAM heatmaps of shape (leads, time)
        patient_id: Optional patient ID for title
        lead_names: Optional list of lead names (I, II, V1, etc.)
    """
    
    # I did this because likely i was thinking that ecg_data, would rely on this error that I see not corresponding to the leaads. 
    ecg_data = ecg_data.permute(1, 0, 2)


    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Squeeze data to remove singleton dimensions
    ecg_data = ecg_data.squeeze(-1).cpu().numpy()
    gradcam_maps = gradcam_maps.cpu().numpy()
    
    num_leads = ecg_data.shape[0]
    time_steps = ecg_data.shape[1]
    
    # Create figure with subplots for each lead
    fig, axes = plt.subplots(num_leads, 1, figsize=(10, 2*num_leads))
    
    # Overall title
    if patient_id:
        fig.suptitle(f'ECG Lead Importance - Patient {patient_id}', fontsize=16)
    
    # Plot each lead with importance overlay
    for i, ax in enumerate(axes):
        # Time axis
        time = np.arange(time_steps)
        
        # Plot ECG signal
        ax.plot(time, ecg_data[i], 'k-', alpha=0.8, linewidth=1.0)
        
        # Create a heatmap background
        ax.imshow(
            gradcam_maps[i][np.newaxis, :],
            aspect='auto',
            extent=[0, time_steps-1, ecg_data[i].min(), ecg_data[i].max()],
            alpha=0.5,
            cmap='jet'
        )
        
        # Set title and labels
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_ylabel('mV')
        
        # Only add x-label for the bottom plot
        if i == num_leads-1:
            ax.set_xlabel('Time (samples)')
    
    plt.tight_layout()
    return fig

def interpret_ecg_model(model, dataset, indices=None):
    """
    Generate and save GradCAM visualizations for specific examples
    
    Args:
        model: ECGNet or ECGCompositeNet model
        dataset: ECG dataset
        indices: Indices of examples to visualize (if None, use first 5)
    """
    if indices is None:
        indices = list(range(min(5, len(dataset))))
    
    for idx in indices:
        # Get sample
        if isinstance(model, ECGCompositeNet):
            ecg_data, label, patient_id = dataset[idx]
            ecg_input = ecg_data.unsqueeze(0)  # Add batch dimension
    
            ecg_input = ecg_input.to('cuda:0') 

            # Generate GradCAM for pre ECG
            pre_gradcam = gradcam_ecg(model, ecg_input, use_post=False)
            fig_pre = visualize_lead_gradcam(
                ecg_input[0, 0],  # First batch, pre ECG
                pre_gradcam[0],   # First batch
                patient_id=f"{patient_id}_pre",
                lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            )
            plt.savefig(f"gradcam_{patient_id}_pre.png", dpi=300, bbox_inches='tight')
            plt.close(fig_pre)
            
            # Generate GradCAM for post ECG
            post_gradcam = gradcam_ecg(model, ecg_input, use_post=True)
            fig_post = visualize_lead_gradcam(
                ecg_input[0, 1],  # First batch, post ECG
                post_gradcam[0],  # First batch
                patient_id=f"{patient_id}_post",
                lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            )
            plt.savefig(f"gradcam_{patient_id}_post.png", dpi=300, bbox_inches='tight')
            plt.close(fig_post)
        else:
            ecg_data, label, patient_id = dataset[idx]
            ecg_input = ecg_data.unsqueeze(0)  # Add batch dimension
            
            # Generate GradCAM
            gradcam_maps = gradcam_ecg(model, ecg_input)
            fig = visualize_lead_gradcam(
                ecg_input[0],      # First batch
                gradcam_maps[0],   # First batch
                patient_id=patient_id,
                lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            )
            plt.savefig(f"gradcam_{patient_id}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Generated GradCAM for patient {patient_id}")


# TODO: Define exactly what I want in an ECG Dataaset and what I am going to do; imagine myself doing it and then go forward with it
"""
Create a new set of activations, say with advice thaat the likely problem is due to overfitting and that the dataset might be too small.

"""

def train_epoch(training_config, model, train_loader, criterion, optimizer, device):
    """
    Train one epoch - DFD does the model have to know about the configuration of the dataset? 
    Yes - It has to know specifically that any* dataset must be a tuple 
    - LT probably wasnt the best option for a dataset interface, but next time i can do that. change to dict?
    - well nothing changes, like there is always data, there is always labels, there is always IDs, 
   -> therefore any dataset MUST output a 3-tuple. 
    
    """
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    for data, labels, _ in train_loader: # training config here has to validate so this fxn can no t worry
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
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    THRESHOLD = 0.5
    predictions_binary = (predictions >= THRESHOLD).astype(float)
    epoch_accuracy = accuracy_score(true_labels, predictions_binary)
    
    return epoch_loss, epoch_auc, epoch_accuracy

class TrainingConfig():
    """
    Class That handles Configuration Management and Instantiation

    Particular Important for handling the mapping and just the handlinlg of creation
    we just wnat the code to just do the doing of making the model, and not having it need to define its own configuraation
    or even validate that the configuration is useful

    Then Training handles telling the functions training what to do and the "env" of training. 
    It will validate different combinations of patient_datasets to the models.

    Acts as a configuration Objective Variable for telling what to do / The ending state + encoding dataset env.
    """

    def __init__(self, dataset = ECGDataset, model = ECGNet): 
        """
        Initialize Parameters

        Args: 
            ECGDataset (Dataset Class) - name of the class to instantiate

        """         
        self.num_epochs = 100
        self.patience = 10
        

        self.dataset_model_configurations = {(ECGDataset, ECGNet), (ECGCompositeDataset, ECGCompositeNet)} 
        self.dataset = dataset
        self.model = model 

        assert isinstance(self.dataset, type) # classes are "type" NOT the clas
        assert isinstance(self.model, type)
        

        self.validate_dataset_model_config(self.dataset, self.model) 


        self.use_post = False
        self.class_weights = torch.Tensor([1.0])
        self.criterion = torch.nn.BCELoss(weight=self.class_weights)  
        self.lr = 0.0001

        self.optimizer_class = torch.optim.Adam 
        self.optimizer = None
    
        self.batch_size = 32
        self.n_workers = 0  

        self.n_splits = 5
        self.device = 'cuda:0'
    
    def validate_dataset_model_config(self, dataset, model):
        """
        Training Config Checks to see if the Dataset is good before telling what to do.
        Checks if the configuration is consistent and found in the valid configurations
        """
        return (dataset, model) in self.dataset_model_configurations
    
    def get_dataset(self, data_dir, label_path): 
        """
        Given dataset object, retrieve a dataset object inside 
        Use another interface for creaing - this might be premature optimization
        # Problem in an OOP class all the variables are very hidden adn the nature of dependencies is not emphasized (feels like i could catch the other varible heuristically)

        Params (Implicit): 
            data_dir - path to the label_dir path
            label_path - path to the labelfile

        

        NOTE:
        Instead of laambdaa - the "params" is the environment" - then here instead of needing to ass it in, i just need to validaate
        that the environment is valoid (nothing given to me, but I still use the env as a function i validate i have all the right paths
        
        I specificaly added data_dir and label_path in this situation because of changing ECG requirement of multiple datasets to input, I needded a better interface way of allowing input of multiple files, to know explicit functional dependencies of information it needs rather than it knowing implicitly
        """

            
 
        
                
        ecg_tensors_pre, filepaths_pre = preprocess_directory(data_dir, get_post=False)
        ecg_tensors_post, filepaths_post = preprocess_directory(data_dir, get_post=True) # Are these deterministic the filepaths? 

        training_data_file_ids_pre = get_patient_ids(filepaths_pre)# helper fucntion due to dependency on tests restricting interface change
        training_data_file_ids_post = get_patient_ids(filepaths_post)
        # if this world were mine 
        
        labels = preprocess_patient_labels(label_path)  # change these so that it aaccomondate
        # Create all objects 
        dataset_pre = ECGDataset(ecg_tensors_pre, training_data_file_ids_pre, labels)
        dataset_post = ECGDataset(ecg_tensors_post, training_data_file_ids_post, labels)
        dataset_composite = ECGCompositeDataset(dataset_pre, dataset_post)  
    
        return_mapping = {(ECGDataset, False): dataset_pre, (ECGDataset, True): dataset_post, (ECGCompositeDataset, True): dataset_composite, (ECGCompositeDataset, False): dataset_composite}

        # filenames are weird --> remove to get only the patient information
        labels = preprocess_patient_labels(label_path) 
        
        return return_mapping[(self.dataset, self.use_post)]

    def generate_model(self): 
        """
        Returns a dictionary containing 4 attributes for training based off of the training config
        """

        model = self.model()
        self.optimizer = self.optimizer_class(model.parameters(), lr=self.lr)

        return model
    def generate_dataloader(self, dataset, split): 
        """
        Generates a dataloader that from indices - (should the indices here be inputted into the function? how does Dataset relate to Config? Should it be controling dataset or fed? 
        """
        dataloader = DataLoader(dataset, 
            batch_size=self.batch_size, 
            sampler=SubsetRandomSampler(split), # notice here that the train variable greatly simplifies the approach (pseudo discovery based)
            num_workers=0
        )

        return dataloader # TODO: Test Dataloader for equivalence cases
    

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
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)   
 
    THRESHOLD = 0.5
    predictions_binary = (predictions >= THRESHOLD).astype(float)
    val_accuracy = accuracy_score(true_labels, predictions_binary)


    return val_loss, val_auc, val_accuracy

def train_model(training_config, model, optimizer, criterion, train_loader, val_loader, device, save_model=True):
    """
    Train a Model To optize towards parameters

    Params:
        model (torch.nn) - model for optimizing
        optimizer (torch.optim) - model for optimizing
        criterion (torch.nn.modules.loss) - 
        train_loader (torch.??) - Datalloader for training data
        val_loader (torch.??) - Dataloader for validation data
        save_model (boolean) - flag to save the best AUC model. Defaults to True

    Returns: (what? I need to be very intentional about how I create the logic)o

    Side Effects: 
        - This will Report Current Training Performance in Epochs to stdout
        - Will write the best modell to /model to torch.state_dict with the follwoing format:
'epoch': epoch,                                                                         - 
    TODO: Get More Description and Organize the Architecture to prevent entropy 
    

    """
    # TODO: Refactor into a Main Config Class #TrainConfig
    num_epochs = training_config.num_epochs 
    best_val_auc = 0
    associated_accuracy = 0   # given best_val_auc, store accuracy associated with it


    best_model_state = None
    patience = training_config.patience# this is something thaat one defines what do to 
    patience_counter = 0
    # load model parameters for validation 

    for epoch in range(num_epochs):
        # Train
        train_loss, train_auc, train_accuracy = train_epoch( 
            training_config,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # Validate - TODO: Turn this into a dictionary to be able to access + get 1 object conceptually, 
        val_loss, val_auc, val_accuracy = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} Val Accuracy: {val_accuracy:.4f}')

    
        # Model selection and early stopping (based on number of times the AUC improves)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            associated_accuracy = val_accuracy 
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            if save_model == False:
                continue

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

    return np.array((best_val_auc, associated_accuracy))



def create_data_splits(dataset, n_splits=5, val_size=0.2):
    """
    Create train/val/test splits that can be extended to k-fold CV.
    Initially uses only one fold but structured for easy k-fold extension.
    
    For dataset only requires informataion about dataset's size AND binary outcome to have equal distribution   
    Args:
        dataset: ECGDataset instance
        n_splits: Number of folds (default=5)
        fold_idx: Which fold to use as test set (default=0)
        val_size: Size of validation set as fraction of training data
    
    Returns:
        list of folds with a dict containing train, val, and test indices
    """
    # Get labels for stratification + splitting
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label.item())
    labels = np.array(labels)
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get indices for the specified fold
    splits = list(skf.split(np.zeros(len(dataset)), labels))
    """
    Splits [([1 2,3], [4, 5, 6]),... ]
    """
    data_splits = [] # difficult to name and specify


    for train_idx, test_idx in splits: 
    
        val_idx = test_idx
  
        data_split =  {
        'train': train_idx,
        'val': val_idx,
        }

        data_splits.append(data_split)
    return data_splits

def do_cross_fold_get_results(training_config, dataset):
    """
    Function that Trains Model on Different Folds of Dataset, returning the Best AUC for each fold. 


    Params:
        dataset (torch.Dataset??) - Pytorch Dataset for training ECG


    Returns: 
        best_aucs (list): list of aucs for each fold created
    """


    best_aucs = []  
    
    splits = create_data_splits(dataset, n_splits=5) # #copied
    
    for split in splits: 
        train_loader = training_config.generate_dataloader(dataset, split['train'])
        val_loader = training_config.generate_dataloader(dataset, split['val'])

        device = training_config.device
        model = training_config.generate_model().to(device)
        class_weights = training_config.class_weights.to(device)
        criterion = training_config.criterion.to(device) 
        optimizer = training_config.optimizer 


        best_auc = train_model(training_config, model, optimizer, criterion, train_loader, val_loader, device, save_model=False)
        best_aucs.append(best_auc)

    # TEMPORARY- added fucntion in this space for gradcam interpretation 
    #interpret_ecg_model(model, dataset)

    return np.array(best_aucs)


def main():
    assert torch.cuda.is_available(), "Error: CUDA required to run trainer.py"

    use_new_dataset = True # very useful because helps with project; it also will not be used further, I considered all optimizations especially for having higher more comprehensive pipelines, but it is not useful
    
    if use_new_dataset == False: 
        data_dir="./data/InterpolatedQRS/"
        label_path="./data/PatientCohort_ECG.xlsx"
    else:
        # I am required to manually preprocess all of these befor
        data_dir="./data/InterpolatedQRS2/"
        label_path="./data/GenCard_ECGs_Jonathan/GenCard_class_labels.xlsx"
                    # maybe here I will just have a separate python module for specifically adapting --> this ismost useful?

    # I had this formaat because, it waas easy to work with / duplicates will not affect, as evidenced by my stable training performance 
    training_config = TrainingConfig(dataset=ECGCompositeDataset, model=ECGCompositeNet)

    #WHYIM[0]: Opened up option for including explicit filepaths, useufl because it allows me to be more explicit on what information it needs to be passed in
    # so that it can operate at the surface (previously before i had no need but now it is useful)
    dataset = training_config.get_dataset(data_dir=data_dir, label_path=label_path)
    



    # TODO: Determine if the validation set is too small? like i want to then just use 2 training sets train and val, no need to leave out test??? --> I want here to get global accuracy metric
    # Create a higher order narrative for interpreting what my code is doing that is consistent and 
    best_aucs = do_cross_fold_get_results(training_config, dataset)
   
    averages =  np.mean(best_aucs, axis=0)
    average_auc, average_accuracy =  averages[0], averages[1]

    print(f"Average AUC: {average_auc}, Average Accuracy: {average_accuracy}")
    breakpoint()

if __name__ == '__main__':
    main()
