clear all
close all
clc

disp("RUN START!")

% Get the full path to the script's folder (current directory)
currentFolder = fileparts(mfilename('fullpath'));

disp("Reading Input File...")
% Construct the full path to the Excel file
inputdat = fullfile(currentFolder, 'Quantification-Test-HPC.xlsx');
quainput = fullfile(currentFolder, 'Quantification-Test-HPC_quares.xlsx');

% Read the data from the Excel file
x1 = readmatrix(inputdat);
xqua = readmatrix(quainput);

disp("Reading Input File Success!")

% Add the function folder path
addpath(fullfile(currentFolder, 'Neural-Network-Function'));

% Display the data to validate readability
% disp('Input Data:')
% disp(x1);

%% DETECTION
disp("Conducting Detection...")
% Call the neural network function
y_det = Detection1440x5v1(x1);
y_det = round(y_det);

%%%%%%%%%%%%%% CONFUSION MATRIX %%%%%%%%%%%%%%%%%%
% Ground truth values (all 1s)
groundTruth = ones(size(y_det));

% Convert predicted and ground truth to binary matrices for plotconfusion
numClasses = 2; % Since we only have binary classes: '0' and '1'
groundTruthBinary = full(ind2vec(groundTruth' + 1, numClasses)); % Convert ground truth to binary matrix
y_det_binary = full(ind2vec(y_det' + 1, numClasses)); % Convert predictions to binary matrix

% Plot the confusion matrix using plotconfusion
plotconfusion(groundTruthBinary, y_det_binary);

% Add title
title('Detection Confusion Matrix');
xlabel('Predicted');
ylabel('True');

% Create the folder if it doesn't exist
if ~exist('charts', 'dir')
   mkdir('charts');
end

% Save the confusion chart as an image file in the folder
saveas(gcf, fullfile('charts', 'plotconfusion_detection_qua_test.png'));



%%%%%%%%%%%%%% ISOLATION PREPARATION %%%%%%%%%%%%%%%%%%%


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
disp("Conducting Isolation...")
% Take the result of the detection out
x2 = combinedfiltered(:, 2:14); %x2 only filled with failed cases
numbered_column2 = combinedfiltered(:, 1);

% Call the neural network function to execute isolation
y_iso = abs(round(Isolation1440x4v1(x2))); %absolute and round altogether

% Replace each row with a one-hot encoding of the maximum value
[numRows, numCols] = size(y_iso);
y_iso_mod = zeros(size(y_iso));
for i = 1:numRows
    [~, colIdx] = max(y_iso(i, :)); % Find index of the maximum value (leftmost in case of ties)
    y_iso_mod(i, colIdx) = 1;          % Set that index to 1
end

% Display the resulting matrix
% disp('Modified Matrix:');
% disp(y_iso_mod);


%%%% CONFUSION MATRIX %%%%%%

% Desired output matrix (target)
desired_output = repmat([0 1 0 0], size(y_iso_mod, 1), 1);

% Convert rows to class labels
[~, predicted_labels] = max(y_iso_mod, [], 2);  % Predicted labels from neural network output
[~, true_labels] = max(desired_output, [], 2); % True labels from desired output

% Convert labels to one-hot encoded matrices for plotconfusion
numClasses = 4; % Number of classes
true_onehot = full(ind2vec(true_labels', numClasses)); % Convert true labels to one-hot
predicted_onehot = full(ind2vec(predicted_labels', numClasses)); % Convert predicted labels to one-hot

% Plot the confusion matrix using plotconfusion
plotconfusion(true_onehot, predicted_onehot);

% Add title
title('Isolation Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');
saveas(gcf, fullfile('charts', 'plotconfusion_isolation_qua_test.png'));



% Display the matrix as an image
% figure;
% imagesc(abs(y_iso)); % Display the matrix as a scaled image
% colormap(gray); % Set the colormap to grayscale
% axis equal tight; % Adjust the axes
% title('Grid Representation of y\_iso');

%Combine the x2 with y_iso
combinedmat2 = [numbered_column2, x2, abs(y_iso_mod)];

% Divide the combined matrix into 4 subsets
iso_lpc = combinedmat2(y_iso_mod(:, 1) == 1, :); % Rows where the LPC column of y_iso is 1
iso_hpc = combinedmat2(y_iso_mod(:, 2) == 1, :); % Rows where the HPC column of y_iso is 1
iso_hpt = combinedmat2(y_iso_mod(:, 3) == 1, :); % Rows where the HPT column of y_iso is 1
iso_lpt = combinedmat2(y_iso_mod(:, 4) == 1, :); % Rows where the LPT column of y_iso is 1

%% Quantification, LPC
disp("Conducting Quantification on:")
disp("• Low Pressure Compressor (LPC)")

% Take the result of the isolation out
x3_lpc = iso_lpc(:, 2:14);
numbered_column3_lpc = iso_lpc(:,1);

y_qua_lpc = Quantification1LPC1440x1v1(x3_lpc);

% disp("Quantification Result (LPC)")
% disp(y_qua_lpc)

combinedmat3_lpc = [numbered_column3_lpc, x3_lpc, y_qua_lpc];

%% Quantification, HPC
disp("• High Pressure Compressor (HPC)")
% Take the result of the isolation out
x3_hpc = iso_hpc(:, 2:14);
numbered_column3_hpc = iso_hpc(:,1);

y_qua_hpc = Quantification2HPC1440x1v1(x3_hpc);

% disp("Quantification Result (HPC)")
% disp(y_qua_hpc)


%%%%%%% COMPARISON AND FIGURE %%%%%%%%%%%%%
FC_HPC = y_qua_hpc(:,1);
nu_HPC = y_qua_hpc(:,2);

figure;
subplot(2, 1, 1)
plot(numbered_column3_hpc, FC_HPC)
hold on
plot(numbered_column, xqua(:,1))
% subplot params
xlim([0 1200]); % Set X-axis limits from 0 to 6
ylim([-7 1]);
%xlabel('Test Case')
ylabel('Flow Degradation (%)')
legend({'Predicted', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

subplot(2, 1, 2)
plot(numbered_column3_hpc, nu_HPC)
hold on
plot(numbered_column, xqua(:,2))
% subplot params
% xlim([0 1200]); % Set X-axis limits from 0 to 6
ylim([-8 1]);
xlabel('Test Case')
ylabel('Flow Efficiency (%)')
legend({'Predicted', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

saveas(gcf, fullfile('charts', 'HPC_comparison.png'));



combinedmat3_hpc = [numbered_column3_hpc, x3_hpc, y_qua_hpc];

%% Quantification, HPT
disp("• High Pressure Turbine (HPT)")


% Take the result of the isolation out
x3_hpt = iso_hpt(:, 2:14);
numbered_column3_hpt = iso_hpt(:,1);

y_qua_hpt =Quantification3HPT1440x1v1(x3_hpt);

% disp("Quantification Result (HPC)")
% disp(y_qua_hpt)

combinedmat3_hpt = [numbered_column3_hpt, x3_hpt, y_qua_hpt];

%% Quantification, LPT
disp("• Low Pressure Turbine (LPT)")
% Take the result of the isolation out
x3_lpt = iso_lpt(:, 2:14);
numbered_column3_lpt = iso_lpt(:,1);

y_qua_lpt = Quantification4LPT1440x1v1(x3_lpt);

% disp("Quantification Result (LPT)")
% disp(y_qua_lpt)





combinedmat3_lpt = [numbered_column3_lpt, x3_lpt, y_qua_lpt];

%% Excel Output

disp("Writing result to Excel...")

% Headers for the sheets (converted to cell array)
headers_A = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN'};
headers_B = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN', 'Detection Result'};
headers_C = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN', 'delta_efficiency', 'Flow Capacity'};

%Combining x1 with case number
A = [numbered_column, x1]; 

% Convert numeric data in A to cell array
data_A = [headers_A; num2cell(A)];
data_B = [headers_B; num2cell(combinedfiltered)];
data_C = [headers_C; num2cell(combinedmat3_lpc)];
data_D = [headers_C; num2cell(combinedmat3_hpc)];
data_E = [headers_C; num2cell(combinedmat3_hpt)];
data_F = [headers_C; num2cell(combinedmat3_lpt)];

% Define the filename
filename = 'NN-Results_qua-test.xlsx';

% Write data_A to the first sheet with headers
writecell(data_A, filename, 'Sheet', 'Input_Case');
writecell(data_B, filename, 'Sheet', 'Detection');
writecell(data_C, filename, 'Sheet', 'Iso-Qua_LPC');
writecell(data_D, filename, 'Sheet', 'Iso-Qua_HPC');
writecell(data_E, filename, 'Sheet', 'Iso-Qua_HPT');
writecell(data_F, filename, 'Sheet', 'Iso-Qua_LPT');

disp("Excel Generation Finished.")
disp("RUN FINISHED!")