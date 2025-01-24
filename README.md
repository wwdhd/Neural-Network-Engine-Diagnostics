# Neural Network Application in Engine Diagnostics
In this study, the Neural Network Machine Learning is used for failure identification in an aircraft engine. In order to execute the identification, the neural network must first detect if the overall engine output classified as failing. Then, it isolates the failing part of the engine. Then, it quantifies the output values of the said part. 

## Parts name:
* The combined_part is the executable MATLAB subroutine, use this if you want to execute the entire neural network or changing the input dataset name
* Neural-Network-Function folder is the neural network parameters folder. Change the file inside if you want to fine tune the result of the parameters.
* The excel file, "Test HPC.xlsx" is the input datasets. Change this for a new engine data

## Checklists

- [x] Make 1 part of the entire neural network
- [X] Make the entire part of the neural network
  - [X] Low Pressure Compressor (LPC)
  - [X] High Pressure Compressor (HPC)
  - [X] High Pressure Turbine (HPT)
  - [X] Low Pressure Turbine (LPT)
- [ ] Create post-processing
  - [X] Excel post-processing (LPT)
  - [ ] Confusion matrix for detection and isolation part
