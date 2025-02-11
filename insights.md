


# Objectives
- Build a Baseline Model such that at some time point

- Evaluates the ECG wave forms 
- Makes a prediction


# Insights to arrive at Baseline Model
- Decide on the architectural changes to build to train model 
    - What is the data format I train --> how to frame information?
    - What kind of network are you proposing? - CNN prior
    - How do I evaluate and show the network what is right? 

## Heuristics for Baseline Model for building
- Get the CE Loss function to measure the subjective "surprisal" to an objective distribution
- build a basic CNN that fits aand processes some data wave form, 
    - when processing i only need to consider questions on the local context only. =
- build with certainty with testing.


### AArchitectural decisions? 
- Bootstrap on this aand see what I need from empimrical experience
### Build environment (Given an dockerfile for running)
```docker build -t ecg_1 . ```# For windows 

# forgot about the -t next time consider every action important




### Run with attatched terminal + local tracking from volume
docker run -v ${PWD}:/app -it ecg_1 bash


## Run assuming that you have the nvidia ctk installed on docker
docker run -gpus all -v ${PWD}:/app -it ecg_1 bash

## Using pytest ini to specificy how to run python files 





[pytest]
pythonpath = . 
Important whe running pytest from the directory need to have pytest ini. 
When runnning the files tells the test files to run as if we are in the project directory. 


## Metacognitvie Conscious Fatigue Objectives
 - Maybe in this idea of consciousness that tension is reallyl just the direction of energy towards structures that manifests in "fatigue"? 
 - then i have to minimize this subjective quality of fatigue and tension in order to be very efficient. "cognitve dissonance" 
 - it is a wasted mental energy.


# Time taken
- 1 pomo - emailing derek + planning
- 1 pomo - learn how to test functinos that are not files + best strategies for getting certianty with low cost. 
- 1 pomo - realized about git errors need for testing --> start with git and realize that I need to learn about managing the broken 
fllow from computer to learn + using the word document. 



# 2-10 
- compoleted the data structure

# 2-11
- TODO - LT complete the new testing fucntionn for the Dataset object
