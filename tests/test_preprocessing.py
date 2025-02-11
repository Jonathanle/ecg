import pytest

import pandas as pd # This is the fileframe
import numpy as np
import tempfile
from pathlib import Path

from trainer import * 
 
import torch
import os

import re
"""
Testing file for testing the functions of the preprocessor
Ensure that this file is within tests/data
and that there exists a directory data/InterpolatedQRS inside
as well as data/misc inside 
"""

@pytest.fixture(scope='module')
def patient_excel_static(): 
    # Added this specifically because existed an action repeated that 
    data_tensor, file_paths = preprocess_directory() # this could be a fixture that stays constant
    
    return data_tensor, file_paths

@pytest.fixture(scope='function')
def ecg_dataset_fixture(patient_excel_static): # fine to have function scope?
    ecg_tensors, filepaths = preprocess_directory()

    training_data_file_ids = get_patient_ids(filepaths)

    labels = preprocess_patient_labels()

    dataset = ECGDataset(ecg_tensors, training_data_file_ids, labels)

    return dataset
@pytest.fixture(scope='function')
def sample_excel(tmppath = "./data/misc"): 
    """Create a temporary Excel file with sample data."""
    # Create sample data
    
    arr = np.zeros((250, 12))


    df = pd.DataFrame(arr) 


    assert df.shape == (250, 12)
    tmppath = Path(tmppath) 
    

    file_path = tmppath / Path("test_data.xlsx")
    df.to_excel(file_path, index = False, header = False)  # does this input the file onn top somehow the headers get squished in # directly relevant
    

    
    yield file_path


    # Clean Up Environment

    if file_path.exists(): 
        os.unlink(file_path) # as
@pytest.fixture(scope='function')
def patient_file(): 
    return './data/InterpolatedQRS/R01_post_anon_interp.xlsx'

def test_excel_import_patient_set_right_dimensions(patient_file): 


    data  = preprocess_data_file(patient_file)

    assert data is not None

    # How do I interface with teh pd with no columns?
    
    assert data.shape == (250, 12)
    assert abs(data[2][5] - 1.74791) < 0.001

def test_excel_import(sample_excel): 

    input_dir = sample_excel 

    data  = preprocess_data_file(input_dir)


    assert data is not None


    # How do I interface with teh pd dataframe?
    
    assert data[0].size == 250
    assert data.shape[1] == 12 
@pytest.mark.parametrize("row,col", [
    (20, 8), 
    (0, 0), 
    (10, 5), 
    (5, 3), 
])
def test_excel_directory_creation(patient_excel_static, row, col):  
    data_tensor, file_paths = patient_excel_static
    
    # VERY IMPORTANT modifications to patient excel static are not desired 


    file_index = 0

    df = pd.read_excel(file_paths[file_index], header=None, index_col=None)
   
    tensor_value = data_tensor[file_index][row][col][0] 
    df_value = df[col][row]

    assert tensor_value == df_value
@pytest.mark.parametrize("row,col", [
    (0,0),
])
def test_excel_directory_creation_metaadata(patient_excel_static, row, col):  # good job on makin the parameters to test more good practice
    data_tensor, file_paths = patient_excel_static

    file_index = 0

    df = pd.read_excel(file_paths[file_index], header=None, index_col=None)
   

    assert len(file_paths) == data_tensor.shape[0]

def test_excel_directory_no_clin_trial(patient_excel_static):

    # Heuristic not as sound or precise (pootential for false positives) but heuristically useful

    _, file_paths = patient_excel_static

    # this one i will learn regex 

 
    for filepath in file_paths: 
        match = re.search(r"Clin", filepath, re.IGNORECASE)

        assert match == None

def test_torch_dataset_loading(patient_excel_static): 
    data_tensor, file_paths = patient_excel_static

    #dataset = ECGDataset(data_tensor, file_paths)

    assert True # pattern to know if it even runs

def test_cuda_is_available_in_env(): 
    assert torch.cuda.is_available()

def test_pre_images_is_default(patient_excel_static):
    # TODO: Analyze how this actually paid off - if i geta hit from not the syntax errors but the core thing it is testing, then if the cost of making now is lower
    # than the perceived errors after it is worth it
    data_tensor, file_paths = patient_excel_static
  
    for filename in file_paths: # error cauught / complexity - file_paths are actually posix paths, if i caught later could have led to issues in interpretaion and printing very worth it it is NOT a string
        assert "R" in filename 

def test_patient_dataset_loading_intermediate(): 
    # Start with the certainty here create the intermediate --> finish --> complete allows building of a code withotu needing to refactor
    # Framework - Bootstrapping the test with the intermediate code - both are built up so that it is easy to verify
    # added header + index col = 0 because cols and rows were labels for rows/columns

    dict_mapping = preprocess_patient_labels()
    

    assert dict_mapping['R03'] == 1
    assert dict_mapping['R04'] == 0

    # assert df['RVEFB']['R03'] == 48.35
def test_patient_data_set_is_int_and_binary(): 
    """
    Learning notes - to test the "classifier" or to meta test, I made the test fail to show that it classifes wrong instances within reason to bootstraap empirica
    confidence DF - get all notes of instances where I find if it does FP or FN. is testing failure then showing TN (it wont false negative) 
    what if the code is bad, and it test true? Problem - False positive - code is bad but it classifies it as good, need to discriminate 
    """
    dict_mapping = preprocess_patient_labels()

    number_test = dict_mapping['R03']

    assert isinstance(number_test,  int) # I verified here from my problem that the "acceptable space" is very insanely constricted + idenfiable
    assert number_test == 1 or number_test == 0

def test_preprocess_patient_labels_only_patient_ids(): 
    dict_mapping = preprocess_patient_labels()
   
    # Function I believe is precise, it only accepts strings greatre than 0 that have R as a key -> this is acceptable anything else is bad. 
    # Work to improve on -- acceptable states beyond R - do you want R xrrwe think about the objective subspace
    for key, value in dict_mapping.items(): 
        assert isinstance(key, str)
        assert len(key) > 0
        assert key[0] == "R" 
       
        # Verify the rest of the number is digits
        assert len(key) >= 2
        assert key[1:].isdigit()

def meta_test_test_patient_dataset_loading_intermediate(): 
    """
    Meta Test for emphsaizeing how certainty is then derived
    in the orignal test I started by looking at the code and having an understandin g
    of what needed to be done

    then this is a classifier test or in other words  "test" test

    This test just assumes nothing about it and test if it is assrting false or true
    assuems the tests themselves are arbritrary and can be unintuitive, but are 
    directly more observable


    Test created to emphasize the nature of a test being able to be black boxed
    as well as th faact that this is a simple test for classifier but is not comprehenvie
    also emphasized is that fact that i use my intuition to bootstrap the certainty 
    so this is cleary overkill, but I will use my intuition most of the time
    """

    # inputs to test: dict mapping output --> False or True
    dict_mapping = {'R03': 1, 'R04': 0} # the function output
    
    eval1 = dict_mapping['R04'] == 0 and dict_mapping['R03'] == 1

    assert eval1 == True


    dict_mapping2 = {'R03': 1, 'R04': 1} # the function output parameter
    eval2 = dict_mapping2['R04'] == 0 and dict_mapping2['R03'] == 1 # function to test


    # testing classifcation to false
    assert eval2 == False # our test for the outside

@pytest.mark.parametrize('filename, patient_id', [
    ('data/interpqrs/R02_asdfasdasdf.xlsx', 'R02'), 
    ('data/interpqrs/R02_asdfasdasdf.xlsx', 'R02'),
    ('reports/R150_quarterly_summary.pdf', 'R150'),
    ('archive/2024/R99_legacy_data_v2.csv', 'R99'),
    ('temp/R47_draft_20240210.txt', 'R47'),
    ('outputs/analysis_R23_final.json', 'R23'),
])
def test_get_patient_id(filename, patient_id): 
    """
    Test Equivalence Cases
    """

    matches = get_patient_id(filename)
    
    assert len(matches) == 1
    assert matches[0] == patient_id
   
def test_get_patient_ids_eq(): 
    filepaths= ['asdfsadf/R01sss.xlsx', 'asdfsadf/R31sss.xlsx', 'asdfsadf/R207sss.xlsx']
    true_patient_ids = ['R01', 'R31', 'R207']

    patient_ids = get_patient_ids(filepaths)

    assert len(patient_ids) == len(filepaths)
    
    # mistake from before used imprecise RXX accepting any digit 
    for patient_id, true_patient_id in zip(patient_ids, true_patient_ids): 
        assert true_patient_id == patient_id

def test_ECG_dataset_training_data_correct_shape(ecg_dataset_fixture): # framework when I change a var name any other variables names I have to change? note to self if i have a ecg fixture as this name
    """
    Idea - I can have shitty tests that has false positives, but get tests that increasingly build 
    on one another to bootstrap if the test caase is too complicaated
    this will false positive everything make it increasingly constricted

    sb - test allows for learning more about exactly how to use a function discovering errors anyway 

    insterad of a loose intuitiuno what *exactly* do you want thefucntion to 

    intuition if i have a test and within a restricted time i do not complete an assignment or deadline then i recognize the testing as not worth it and strictly defined as "inefficient" with some deadilone

    time spent - 1hr completing: probalbly less cost than the errors in all time later
    """

    dataset = ecg_dataset_fixture
    
    first_patient_ecg, outcome, patient_id = dataset[0]
    
    # sb this one is unce4tain, need to bootstrap with my learning pdb experiment -- VERFIED TRUE
    #assert isinstance(first_patient_ecg, np.ndarray) # must be a np.array? no we want a torch tensor

    assert isinstance(first_patient_ecg, torch.Tensor) 
    assert isinstance(outcome, torch.Tensor)
    assert isinstance(patient_id, str) # I know this is correct but this is documentation / monitors regression hits (find if it actually is valuable) I know but I stay conservative

    assert first_patient_ecg.dtype == torch.float32 # what should this be ? bootstrap from pdb, 
    assert first_patient_ecg.shape == torch.Size([250, 12, 1]) # exactly right uncertain if test well
    assert outcome.dtype == torch.int64 # Advice from Claude to standardize for CE Loss
    assert outcome.shape == torch.Size([1]) # exactly right uncertain if test well


    # test if the string has correct id using knowledge of re

    pattern = re.compile(r"R\d+") # need to add \d not d+, use recompile to standardize pattern 
    # error add a r

    matches = pattern.match(patient_id) #NOT re.find re.match need to add the test str
    # match specifically starts from the beginning as a strict reuiqrement from searchk

    assert matches is not None
     

    


if __name__ == '__main__':
    sample_excel()
