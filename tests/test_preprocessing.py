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

    dataset = ECGDataset(data_tensor, file_paths)

    assert True # pattern to know if it even runs

def test_cuda_is_available_in_env(): 
    assert torch.cuda.is_available()

def test_pre_images_is_default(patient_excel_static):
    # TODO: Analyze how this actually paid off - if i geta hit from not the syntax errors but the core thing it is testing, then if the cost of making now is lower
    # than the perceived errors after it is worth it
    data_tensor, file_paths = patient_excel_static
  
    for filename in file_paths: # error cauught / complexity - file_paths are actually posix paths, if i caught later could have led to issues in interpretaion and printing very worth it it is NOT a string
        assert "pre" in filename 


if __name__ == '__main__':
    sample_excel()
