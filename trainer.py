
import pandas as pd
from pathlib import Path

def preprocess_data(input_dir = "./data/InterpolatedQRS"):
    """
    Given unknown / stochastic environment, process all of the patient 
    files to get all of the relevant data --> ensure that this is completely processed

    
    Format of the directory - 
    All patient files are in input_dir
    each patient is associated with a xlsx file


    format - patient folder / 
    (PATIENT, TIME, LEAD, Voltage)

    # TODO: Figure out how to test this code so that I obtain 100% certainty to build models.
    """
    # List all file paths ending in xlsx


    filename = "R01_pre_anon_interp.xlsx"
    # TODO: Import the dataset as a pandaqs 

    df = pd.read_excel(input_dir)

    return df



def main():
    # preprocess data --> Tensor 
    # How is the information objectively structured? 
    # TODO: create an process that later on
    # traansforms the data into an interface.





    # define model
    # model train on data.  
    

    return
if __name__ == '__main__':
    main()
