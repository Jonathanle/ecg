import pytest

import pandas as pd # This is the fileframe
import numpy as np
import tempfile
from pathlib import Path

from trainer import * 
 
import torch
import os

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


def test_get_patient_id(): 
    filename = "R02_asdfasdfafdasfasfd.xlsx"
    # patient_id = "asdfasdf"# get_patient_id(filename) wrote this test to fail initially

    patient_id = get_patient_id(filename)

    assert patient_id == "R02"
    

if __name__ == '__main__':
    sample_excel()
