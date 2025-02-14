"""
TODO: Create a entropy reducing infrastructure for pytorch code

1. Determine exactly what I want to be testing 
    - My subjective belief on importing tensors work
    - I can test by writing out the code by myself (
    - gradient knowledge computation
"""


import pytest 

from trainer import * # DF imprecise need better precision ono what i want 

import torch



@pytest.fixture(scope='module')
def patient_dataset():
    ecg_tensors, filepaths = preprocess_directory(get_post=True)

    training_data_file_ids = get_patient_ids(filepaths)# helper fucntion due to dependency

    # filenames are weird --> remove to get only the patient information
    labels = preprocess_patient_labels() # change to a dictionary


    dataset = ECGDataset(ecg_tensors, training_data_file_ids, labels)

    return dataset

@pytest.fixture(scope='module')
def model():
    return ECGNet()

@pytest.fixture(scope='module')
def training_config(patient_dataset, model):    
    
    train_config = TrainingConfig(model=model, dataset=patient_dataset)  
    return train_config

def test_cuda_is_available():
    assert torch.cuda.is_available()



def test_tensor_is_cpu():
    """
    Test to Reduce the entropy of my understanding of what needs to be in a tensor

    Tensors will always be on a cpu
    # Surprise 1 - Tensor vs tensor i shoudl use tensor heuristically
    """
    tensor = torch.tensor([1, 2], dtype=torch.float32)

    assert tensor.device == torch.device('cpu')
    assert tensor.dtype == torch.float32

def test_tensor_is_in_cuda():
    """
    Test ability to get a tensor in devices 

    Surprise - torch.device('cuda') vs torch.device('cuda', index=0)
    indexes represent which "gpu" to compute something - df parallel processing
    
    torch.device('cuda') === torch.device('cuda:0') === torch.device('cuda', index=0)

    # in future - use cuda:0 ---> torch.device equates this with torch.device('cuda', index=0)

    """

    tensor = torch.tensor([1, 2], dtype=torch.float32).to(torch.device('cuda')) # cud

    # assumed = is equal to ==
    assert tensor.device == torch.device('cuda:0')


def test_loss_item():
    """
    Why is detach important?
    """


    tensor = torch.tensor([1], dtype=torch.int64)

    python_number = tensor.item()

    assert tensor.dtype == torch.int64
    assert isinstance(python_number, int)

def test_exception_loss_item():
    """
    Showing that i can test an exception occuring and then evaluate that to true + tensors .item only accepts tensors that are scalar

    This flow is conventional and arbritrary 
    """

    tensor = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(RuntimeError) as exc_info:
        python_number = tensor.item()

    assert str(exc_info.value) == "a Tensor with 2 elements cannot be converted to Scalar"

def test_CV_split_len_5(patient_dataset):
    """
    Show that StratifiedKFold splits datasets into equal proportions of sample
    maybe these entropy can be temporararily corrected via using breakpoint() 

    Show that create_data_split splits it into 5 files
    """
    data_splits = create_data_splits(patient_dataset, n_splits=5)

    assert len(data_splits) == 5

def test_CV_split_has_dict(patient_dataset): 
    """
    Testing Function to describe if the format of each split is a dict
    """
    data_splits = create_data_splits(patient_dataset, n_splits=5)

    for item in data_splits: 
        assert isinstance(item, dict)
        assert 'train' in item
        assert 'test' in item
        assert 'val' in item

def test_CV_split_equal_split(patient_dataset): 
    """
    Testing Function to describe if the format of each split is a dict

    - equal split
    - no zero length split - never an acceptable state where we have test sets.
    """
    data_splits = create_data_splits(patient_dataset, n_splits=5)
    
    patient_dataset_len = len(patient_dataset)

    for data_split in data_splits: 
        train_set = data_split['train']
        val_set = data_split['test']
        test_set = data_split['val']
    
        total_len = len(train_set) + len(val_set) + len(test_set)
       
        """
        Learn about deriving information from events -> this was directly made from an integrated system test that the length of the dataloader did not work

        - very different than subjective anticipation of errors

        """
        assert len(train_set) != 0 
        assert len(val_set) != 0 
        assert len(test_set) != 0

        assert total_len == patient_dataset_len
def test_TrainingConfig_can_genereate_model(training_config):
        
    model = training_config.generate_model()
    assert isinstance(model, ECGNet)
def test_TrainingConfig_can_generate_dataset(training_config):
        
    dataset = training_config.get_dataset()

    assert len(dataset[0]) == 3 #heuristic for a good dataset

def test_TrainingConfig_can_generate_dataloader(training_config):
     
    dataset = training_config.get_dataset()
    dataloader = training_config.generate_dataloader(dataset, [0, 1, 10])
    

    # how to verify ana analyze dataloader's state?

    assert len(dataloader.sampler.indices) == 3 # next time for functions i know i need to learn about its state from the documethataion via asking claude
    assert True





        
