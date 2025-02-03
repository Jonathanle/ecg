# ecg




## Development Specific Ideaas on Building
- python -m -- add the directory to searching for modules? 
- this is done because then when we run files in pytest, when we run the code, python is trying to import modules
- python then in this case, is trying to search for those modules imported.



## Learned Arbritrary Documentation



### Build environment (Given an dockerfile for running)
docker build -t ecg_1 .




### Run with attatched terminal + local tracking from volume
docker run -v ${PWD}:/app -it ecg_1 bash



## Using pytest ini to specificy how to run python files 


[pytest]
pythonpath = . 
Important whe running pytest from the directory need to have pytest ini. 
When runnning the files tells the test files to run as if we are in the project directory. 