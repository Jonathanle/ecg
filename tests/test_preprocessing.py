import pytest

import pandas as pd # This is the fileframe
import tempfile
from pathlib import Path


# Figure out how to create a testing script. For testing files

def sample_excel(): 
    """Create a temporary Excel file with sample data."""
    # Create sample data
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'salary': [50000, 60000, 75000, 65000],
        'experience': [3, 5, 8, 4]
    }
    df = pd.DataFrame(data)


    pd.to_excel("../data/misc", index = True) 


if __name__ == '__main__':
    sample_excel()
