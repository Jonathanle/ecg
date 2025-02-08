import pytest

import pandas as pd # This is the fileframe
import tempfile
from pathlib import Path

from trainer import preprocess_data

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



def test_excel_import(sample_excel): 

    input_dir = sample_excel 

    data  = preprocess_data(input_dir)


    assert data is not None


if __name__ == '__main__':
    sample_excel()
