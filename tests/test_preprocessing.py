import pytest

import pandas as pd # This is the fileframe
import tempfile
from pathlib import Path

from trainer import * 

import os


# Figure out how to create a testing script. For testing files

@pytest.fixture(scope='function')
def sample_excel(tmppath = "./data/misc"): 
    """Create a temporary Excel file with sample data."""
    # Create sample data
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'salary': [50000, 60000, 75000, 65000],
        'experience': [3, 5, 8, 4]
    }
    df = pd.DataFrame(data)

    tmppath = Path(tmppath) 
    

    file_path = tmppath / Path("test_data.xlsx")
    df.to_excel(file_path, index = True) 
    
    yield file_path


    # Clean Up Environment

    if file_path.exists(): 
        os.unlink(file_path) # as
@pytest.fixture(scope='function')
def patient_file(): 
    return './data/InterpolatedQRS/R110_pre_anon_interp.xlsx'

def test_excel_import_patient_set_right_dimensions(patient_file): 


    data  = preprocess_data_file(patient_file)

    assert data is not None

    # How do I interface with teh pd with no columns?
    
    assert data.shape == (250, 12)

def test_excel_import(sample_excel): 

    input_dir = sample_excel 

    data  = preprocess_data_file(input_dir)


    assert data is not None


    # How do I interface with teh pd dataframe?
    
    assert data['name'].size == 4
    assert data['salary'][2] == 75000

if __name__ == '__main__':
    sample_excel()
