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
y_det = Detection1440x5v1(x1);

% Display the output
disp('Detection Output:');
round(y_det)

% Combined the result of the detection with the result of the detection
combinedmat = [x1, round(y_det)];

% Create a filtered matrix, for only the failed detection cases
% Initialize an empty array to store rows where the first element is < 5
combinedfiltered = [];

%Filter the rows with loops
for i = 1:size(combinedmat, 1)
    if combinedmat(i, 14) > 0
        combinedfiltered = [combinedfiltered; combinedmat(i, :)];  % Append the entire row to B if the first element is = 1
    end
end

%disp(combinedfiltered);

%% ISOLATION

% Take the result of the detection out
x2 = combinedfiltered(:, 1:13); %x2 only filled with failed cases

% Call the neural network function to execute isolation
y_iso = round(Isolation1440x4v1(x2));

% Replace 2 and 3 (positive and negative) with 0, 
% because anything above 0 is considered affected by the part that has
% value of 1

y_iso(abs(y_iso) == 2 | abs(y_iso) == 3) = 0;

% Display the modified matrix
disp(abs(y_iso));

%Combine the x2 with y_iso
combinedmat2 = [x2, abs(y_iso)];

% Divide the combined matrix into 4 subsets
iso_lpc = combinedmat2(y_iso(:, 1) == 1, :); % Rows where the LPC column of y_iso is 1
iso_hpc = combinedmat2(y_iso(:, 2) == 1, :); % Rows where the HPC column of y_iso is 1
iso_hpt = combinedmat2(y_iso(:, 3) == 1, :); % Rows where the HPT column of y_iso is 1
iso_lpt = combinedmat2(y_iso(:, 4) == 1, :); % Rows where the LPT column of y_iso is 1

%% Quantification, LPC

% Take the result of the isolation out
x3_lpc = iso_lpc(:, 1:13);

y_qua = Quantification1IPC1440x1v1(x3_lpc);

disp("Quantification Result (LPC)")
disp(y_qua)

