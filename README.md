# Neural Network Application in Engine Diagnostics
In this study, the Neural Network Machine Learning is used for failure identification in an aircraft engine. In order to execute the failure identification and quantification, the algorithm will do as such:
1. DETECTION: The first neural network must detect if the overall engine can be classified as failing. Then, the cases in which the engine is not failing is excluded from the calculation, ensuring the efficiency and effectivity of the calculation.
2. ISOLATION: The failing cases are then specifically analysed in terms of the failing part using the second neural network algorithm, whether it is LPC, HPC, HPT, or LPT. The sorting algorithm then put the cases into specific failure groups (LPC failure groups, HPC failure groups, etc.)
3. QUANTIFICATION: The third neural network algorithm quantifies the output values of every failure groups in terms of the change of efficiency (delta_efficiency) and flow capacity change, both compared in percentage of a normal and ideal engine.

## Parts name:
* The excel file, `Test HPC.xlsx` and `Quantification-Test-HPC.xlsx` are the input datasets. The first one is a small dataset, while the latter has larger dataset. Change this for a new engine data
* Excel file, `Quantification-Test-HPC_quares.xlsx` is the actual dataset for quantification result comparison
* `\Neural-Network-Function\` folder is the neural network parameters folder. Change the file inside if you want to fine tune the result of the parameters
* The `combined_part.m` is the executable MATLAB subroutine, use this if you want to execute the entire neural network or changing the input dataset name
* `\chart\` folder is the graphics/visualisation result of the MATLAB subroutine
* `NN-results.xslx` is the excel post-processing result of the diagnostics. It shows the sorted results of detection, isolation+quantification data

## Features
1. Neural Network Analysis on Detection, Isolation, and Quantification
2. Post-processing, including (on the order):
   1. Confusion matrix for detection and isolation phase
   2. Visualisation for the quantification part for both flow capacity and efficiency
   3. Statistics for flow capacity and efficiency result: Range, Mean, Standard Deviation, Errors (L1 and L2), 3-sigmas
   4. Microsoft Excel output for all of the calculations

The NN parameters were made using Deep Learning Toolbox in MATLAB, and the input data is generated using Pythia, an in-house application made by Cranfield University.
