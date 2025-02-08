
import pandas as pd
from pathlib import Path

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

    df = pd.read_excel(filepath, header = None)

    assert df.shape == (NUM_TIME_STEPS, NUM_LEADS) 


    return df


def preprocess_directory(input_dir = "./data/InterpolatedQRS/"):
    """
    Processes all the xlsx files and returns a group of the pytorch
    datasets
    """

    return

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
