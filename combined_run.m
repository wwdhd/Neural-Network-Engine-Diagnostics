clear all
close all
clc

% Path to your Excel file
filename = 'Test HPC.xlsx';

% Change the xlsx data into matrix
x1 = readmatrix(filename);

% Display the data to validate the readability
disp('Input Data')
disp(x1);


%% DETECTION
% Call the neural network function
y_det = myNeuralNetworkFunctionHPT(x1);

% Display the output
disp('Detection Output:');

combinedmat = [x1, round(y_det)];

% Initialize an empty array to store rows where the first element is < 5
combinedfiltered = [];

% Loop through each row of the array
for i = 1:size(combinedmat, 1)
    if combinedmat(i, 14) > 0
        combinedfiltered = [combinedfiltered; combinedmat(i, :)];  % Append the entire row to B if the first element is = 1
    end
end

%disp(combinedfiltered);

%% ISOLATION
x2 = combinedfiltered(:, 1:13);

% Call the neural network function
y_iso = myNeuralNetworkFunctionHPCHPTIso(x2);

% Display the output
disp('Isolation Output:');
disp(round(y_iso));

%if HOT = 1, maka ambil

%% QUANTIFICATION
% Call the neural network function
y_qua = myNeuralNetworkFunctionHPTQua(x1);

% Display the output
disp('Quantification Output:');
disp(y_qua);