
# TODO: Create an interpretability mechanism for helping to identify what the ECG was looking at for an interpolatedQRS dataset 




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


# atrohp or dedialate 


# purpose of post
- the long term obserbalb eoutcome is diferent post is still used 
- long term observable outcome. 

- post ecg signasl are not "hindsight" 
 - ecg signals show the outcome? property observed much later not after treatment


# Tasks on validation on the model  2/11/2025
- verifyt
- Ensemble 
- Pre CRT for the response 
- Post CRT
- BE RIGOROUS  - use the validation set, use testing to make a very clear picture going on it should be invisible.
--> ensemble

# DF How can I interpret a model ?
- Interpretability


# Phd Development - It really is inherent stuff done allot of the load is the obtaining of certainty and full clarity this is high cost.
- all of the small stuff.
- With entropy a lot of the stuff.
- invest energy thata minimizes entropy
- phd simulation


# 5fd 
nightmarkes of phds solutions -> create
- I need to then knoww and see the probelm space; then i need to then define the objectives and see if it is even worthy. (Evaluate) 





# 4/2 How do I change the ECG DAtaset so that I can adapt towards the preprocessing code? 

In the trainer file, it seems asa though I end up using the dataset, object. This get_dataset object is importnat because it helps me return the dataset that I need in order to make it, but it seems as though it is very dependent on the fact that I have a "pre" and a "post". It seems too hard to change, maybe it seems to point to the fact that II should just create an adapter that allows on the outside to have a new format --> much more useful, i can just adapt the code, and then stil use the very useful code, less heavy than just editing the structure


# Here, because I have a lot of data, the next logical progression is a formalization of a preprocessing pipeline for any kind of dataset; here i am just using the adapter to stay very flexible.
