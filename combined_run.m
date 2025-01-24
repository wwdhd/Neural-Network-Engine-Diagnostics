clear all
close all
clc

% Get the full path to the script's folder (current directory)
currentFolder = fileparts(mfilename('fullpath'));

% Construct the full path to the Excel file
filename = fullfile(currentFolder, 'Test HPC.xlsx');

% Read the data from the Excel file
x1 = readmatrix(filename);

% Add the function folder path
addpath(fullfile(currentFolder, 'Neural-Network-Function'));

% Display the data to validate readability
disp('Input Data:')
disp(x1);


%% DETECTION
% Call the neural network function
y_det = Detection1440x5v1(x1);

% Display the output
disp('Detection Output:');
round(y_det)

% Combined the result of the detection with the result of the detection
numbered_column = (1:size(y_det, 1))'; %Creating the case number for the ease of analysis
combinedmat = [numbered_column, x1, round(y_det)];


% Create a filtered matrix, for only the failed detection cases
% Initialize an empty array to store rows where the first element is < 5
combinedfiltered = [];

%Filter the rows with loops
for i = 1:size(combinedmat, 1)
    if combinedmat(i, 15) > 0
        combinedfiltered = [combinedfiltered; combinedmat(i, :)];  % Append the entire row to B if the first element is = 1
    end
end


%disp(combinedfiltered);

%% ISOLATION

% Take the result of the detection out
x2 = combinedfiltered(:, 2:14); %x2 only filled with failed cases
numbered_column2 = combinedfiltered(:, 1);

% Call the neural network function to execute isolation
y_iso = round(Isolation1440x4v1(x2));

% Replace 2 and 3 (positive and negative) with 0, 
% because anything above 0 is considered affected by the part that has
% value of 1

y_iso(abs(y_iso) == 2 | abs(y_iso) == 3) = 0;

% Display the modified matrix
disp(abs(y_iso));

% Display the matrix as an image
% figure;
% imagesc(abs(y_iso)); % Display the matrix as a scaled image
% colormap(gray); % Set the colormap to grayscale
% axis equal tight; % Adjust the axes
% title('Grid Representation of y\_iso');

%Combine the x2 with y_iso
combinedmat2 = [numbered_column2, x2, abs(y_iso)];

% Divide the combined matrix into 4 subsets
iso_lpc = combinedmat2(y_iso(:, 1) == 1, :); % Rows where the LPC column of y_iso is 1
iso_hpc = combinedmat2(y_iso(:, 2) == 1, :); % Rows where the HPC column of y_iso is 1
iso_hpt = combinedmat2(y_iso(:, 3) == 1, :); % Rows where the HPT column of y_iso is 1
iso_lpt = combinedmat2(y_iso(:, 4) == 1, :); % Rows where the LPT column of y_iso is 1

%% Quantification, LPC

% Take the result of the isolation out
x3_lpc = iso_lpc(:, 2:14);
numbered_column3_lpc = iso_lpc(:,1);

y_qua_lpc = Quantification1LPC1440x1v1(x3_lpc);

disp("Quantification Result (LPC)")
disp(y_qua_lpc)

combinedmat3_lpc = [numbered_column3_lpc, x3_lpc, y_qua_lpc];

%% Quantification, HPC

% Take the result of the isolation out
x3_hpc = iso_hpc(:, 2:14);
numbered_column3_hpc = iso_hpc(:,1);

y_qua_hpc = Quantification2HPC1440x1v1(x3_hpc);

disp("Quantification Result (HPC)")
disp(y_qua_hpc)

combinedmat3_hpc = [numbered_column3_hpc, x3_hpc, y_qua_hpc];

%% Quantification, HPT

% Take the result of the isolation out
x3_hpt = iso_hpt(:, 2:14);
numbered_column3_hpt = iso_hpt(:,1);

y_qua_hpt =Quantification3HPT1440x1v1(x3_hpt);

disp("Quantification Result (HPC)")
disp(y_qua_hpt)

combinedmat3_hpt = [numbered_column3_hpt, x3_hpt, y_qua_hpt];

%% Quantification, LPT

% Take the result of the isolation out
x3_lpt = iso_lpt(:, 2:14);
numbered_column3_lpt = iso_lpt(:,1);

y_qua_lpt = Quantification4LPT1440x1v1(x3_lpt);

disp("Quantification Result (LPT)")
disp(y_qua_lpt)

combinedmat3_lpt = [numbered_column3_lpt, x3_lpt, y_qua_lpt];

%% Excel Output

% Headers for each sheet (converted to cell array)
headers_A = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN'};
headers_B = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN', 'Detection Result'};
headers_C = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN', 'delta_efficiency', 'Flow Capacity'};

% Assuming A is already defined as a numeric matrix
A = [numbered_column, x1];  % Define A based on your data

% Convert numeric data in A to cell array
data_A = [headers_A; num2cell(A)];
data_B = [headers_B; num2cell(combinedfiltered)];
data_C = [headers_C; num2cell(combinedmat3_lpc)];
data_D = [headers_C; num2cell(combinedmat3_hpc)];
data_E = [headers_C; num2cell(combinedmat3_hpt)];
data_F = [headers_C; num2cell(combinedmat3_lpt)];

% Define the filename
filename = 'NN-Results.xlsx';

% Write data_A to the first sheet with headers
writecell(data_A, filename, 'Sheet', 'Input_Case');
writecell(data_B, filename, 'Sheet', 'Detection');
writecell(data_C, filename, 'Sheet', 'Iso-Qua_LPC');
writecell(data_D, filename, 'Sheet', 'Iso-Qua_HPC');
writecell(data_E, filename, 'Sheet', 'Iso-Qua_HPT');
writecell(data_F, filename, 'Sheet', 'Iso-Qua_LPT');




