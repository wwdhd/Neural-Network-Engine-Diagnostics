# Neural Network Application in Engine Diagnostics
In this study, the Neural Network Machine Learning is used for failure identification in an aircraft engine. In order to execute the failure identification and quantification, the algorithm will do as such:
1. DETECTION: The first neural network must detect if the overall engine can be classified as failing. Then, the cases in which the engine is not failing is excluded from the calculation, ensuring the efficiency and effectivity of the calculation.
2. ISOLATION: The failing cases are then specifically analysed in terms of the failing part using the second neural network algorithm, whether it is LPC, HPC, HPT, or LPT. The sorting algorithm then put the cases into specific failure groups (LPC failure groups, HPC failure groups, etc.)
3. QUANTIFICATION: The third neural network algorithm quantifies the output values of every failure groups in terms of the change of efficiency (delta_efficiency) and flow capacity change, both compared in percentage of a normal and ideal engine.

## Parts name:
* The excel file, `Test HPC.xlsx` is the input datasets. Change this for a new engine data
* `\Neural-Network-Function\` folder is the neural network parameters folder. Change the file inside if you want to fine tune the result of the parameters
* The `combined_part.m` is the executable MATLAB subroutine, use this if you want to execute the entire neural network or changing the input dataset name
* `\chart\` folder is the graphics/visualisation result of the MATLAB subroutine
* `NN-results.xslx` is the excel post-processing result of the diagnostics. It shows the sorted results of detection, isolation+quantification data


## Checklists

- [x] Make 1 part of the entire neural network
- [X] Make the entire part of the neural network
  - [X] Low Pressure Compressor (LPC)
  - [X] High Pressure Compressor (HPC)
  - [X] High Pressure Turbine (HPT)
  - [X] Low Pressure Turbine (LPT)
- [ ] Create post-processing
  - [X] Excel post-processing
  - [X] Confusion matrix for detection and isolation part
  - [ ] Flow Prediction for quantification part
