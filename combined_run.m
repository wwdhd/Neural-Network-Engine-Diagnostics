%%%%%%%%%%%%%% | NEURAL NETWORK FOR ENGINE DIAGNOSTICS | %%%%%%%%%%%%%
clear all
close all
clc

disp("RUN START!")

% Get the full path to the script's folder (current directory)
currentFolder = fileparts(mfilename('fullpath'));

disp("Reading Input File...")
% Construct the full path to the Excel file
inputdat = fullfile(currentFolder, 'isotest_input.xlsx');
isoinput = fullfile(currentFolder, 'isotest.xlsx');
quainput_lpc = fullfile(currentFolder, 'isotest_quares_lpc.xlsx');
quainput_hpc = fullfile(currentFolder, 'isotest_quares_hpc.xlsx');
quainput_hpt = fullfile(currentFolder, 'isotest_quares_hpt.xlsx');
quainput_lpt = fullfile(currentFolder, 'isotest_quares_lpt.xlsx');

% Read the data from the Excel file
x1 = readmatrix(inputdat);
xiso = readmatrix(isoinput);
xqua_lpc = readmatrix(quainput_lpc);
xqua_hpc = readmatrix(quainput_hpc);
xqua_hpt = readmatrix(quainput_hpt);
xqua_lpt = readmatrix(quainput_lpt);

disp("Reading Input File Success!")

% Add the function folder path
addpath(fullfile(currentFolder, 'Neural-Network-Function'));

%% %%%%%%% DETECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Conducting Detection...")
% Call the neural network function
y_det = CLASS1_C(x1);
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
xlabel('Target Class');
ylabel('Output Class');

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

combinedfiltered = combinedmat;

% % Create a filtered matrix, for only the failed detection cases
% % Initialize an empty array to store rows where the first element is < 5
% combinedfiltered = [];
% 
% %Filter the rows with loops
% for i = 1:size(combinedmat, 1)
%     if combinedmat(i, 15) > 0
%         combinedfiltered = [combinedfiltered; combinedmat(i, :)];  % Append the entire row to B if the first element is = 1
%     end
% end


%disp(combinedfiltered);

%% %%%%%%% ISOLATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Conducting Isolation...")
% Take the result of the detection out
x2 = combinedfiltered(:, 2:14); %x2 only filled with failed cases
numbered_column2 = combinedfiltered(:, 1);

% Call the neural network function to execute isolation
y_iso = abs(round(CLASS2_A(x2))); %absolute and round altogether

% Replace each row with a one-hot encoding of the maximum value
[numRows, ~] = size(y_iso);
y_iso_mod = zeros(size(y_iso));
for i = 1:numRows
    [~, colIdx] = max(y_iso(i, :)); % Find index of the maximum value (leftmost in case of ties)
    y_iso_mod(i, colIdx) = 1;          % Set that index to 1
end


%%%% CONFUSION MATRIX %%%%%%

% Convert rows to class labels
[~, predicted_labels] = max(y_iso_mod, [], 2);  % Predicted labels from neural network output
[~, true_labels] = max(xiso, [], 2); % True labels from desired output

% Convert labels to one-hot encoded matrices for plotconfusion
numClasses = 4; % Number of classes
true_onehot = full(ind2vec(true_labels', numClasses)); % Convert true labels to one-hot
predicted_onehot = full(ind2vec(predicted_labels', numClasses)); % Convert predicted labels to one-hot

% Plot the confusion matrix using plotconfusion
plotconfusion(true_onehot, predicted_onehot);

% Add title
title('Isolation Confusion Matrix');
xlabel('Target Class');
ylabel('Output Class');
saveas(gcf, fullfile('charts', 'plotconfusion_isolation_qua_test.png'));

%Combine the x2 with y_iso
combinedmat2 = [numbered_column2, x2, abs(y_iso_mod)];

% Divide the combined matrix into 4 subsets
iso_lpc = combinedmat2(y_iso_mod(:, 1) == 1, :); % Rows where the LPC column of y_iso is 1
iso_hpc = combinedmat2(y_iso_mod(:, 2) == 1, :); % Rows where the HPC column of y_iso is 1
iso_hpt = combinedmat2(y_iso_mod(:, 3) == 1, :); % Rows where the HPT column of y_iso is 1
iso_lpt = combinedmat2(y_iso_mod(:, 4) == 1, :); % Rows where the LPT column of y_iso is 1

%% %%%%%%% Quantification, LPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Conducting Quantification on:")
disp("• Low Pressure Compressor (LPC)")

% Take the result of the isolation out
x3_lpc = iso_lpc(:, 2:14);
numbered_column3_lpc = iso_lpc(:,1);

y_qua_lpc = APPROX1_C(x3_lpc);

%% %%%%% COMPARISON AND FIGURE %%%%%%%%%%%%%
FC_lpc = y_qua_lpc(:,1);
eff_lpc = y_qua_lpc(:,2);


% Plot 1: HPC_comparison_FlowDeg
% Normalise the numbered column
numbered_column_xqua_lpc = (1:size(xqua_lpc, 1))' + numbered_column3_lpc(1);

figure;
plot(numbered_column3_lpc, (FC_lpc))
hold on
plot(numbered_column_xqua_lpc, (xqua_lpc(:,1)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-7.5 1]);
xlabel('Dataset')
ylabel('Flow Capacity (%)')
legend({'Output', 'Target'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
 saveas(gcf, fullfile('charts', 'LPC_comparison_FlowDeg.png'));

% Plot 2: HPC_comparison_FlowDeg
figure;
plot(numbered_column3_lpc, (eff_lpc))
hold on
plot(numbered_column_xqua_lpc, (xqua_lpc(:,2)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-7.5 1.5]);
xlabel('Dataset')
ylabel('Efficency (%)')
legend({'Output', 'Target'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Set font to Times New Roman
%set(gca, 'FontName', 'Times New Roman');

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'LPC_comparison_FlowEff.png'));

%%%%%% PERFORMANCE STATISTICS %%%%%

% Calculate the range, mean, and standard deviation for array A (FC)
range_FC_lpc = range(FC_lpc);
mean_FC_lpc = mean(FC_lpc);
std_FC_lpc = std(FC_lpc);

% Range Mean StdDev for array B (Eff)
range_eff_lpc = range(eff_lpc);
mean_eff_lpc = mean(eff_lpc);
std_eff_lpc = std(eff_lpc);

% Error
if size(FC_lpc, 1) > size(xqua_lpc,1)
    FC_lpc = FC_lpc(1:size(xqua_lpc,1), :);
    eff_lpc = eff_lpc(1:size(xqua_lpc,1), :);
    
    error_fc_lpc = FC_lpc - xqua_lpc(:,1);
    error_eff_lpc = eff_lpc - xqua_lpc(:,2);

elseif size(FC_lpc, 1) < size(xqua_lpc,1)
    xqua_trunc_1_lpc = xqua_lpc(1:size(FC_lpc,1), 1);
    xqua_trunc_2_lpc = xqua_lpc(1:size(eff_lpc,1), 2);
    
    error_fc_lpc = FC_lpc - xqua_trunc_1_lpc;
    error_eff_lpc = eff_lpc - xqua_trunc_2_lpc;
else
    error_fc_lpc = FC_lpc - xqua_lpc(:,1);
    error_eff_lpc = eff_lpc - xqua_lpc(:,1);
end

% L1 Norm (Sum of absolute differences)
L1_fc_lpc = sum(abs(error_fc_lpc)) / sum(abs(FC_lpc)) * 100;
L1_eff_lpc = sum(abs(error_eff_lpc)) / sum(abs(eff_lpc)) * 100;

% L2 Norm (Square root of sum of squared differences)
L2_fc_lpc = sqrt(mean(error_fc_lpc.^2)) / (mean(abs(xqua_lpc(:,1)))) * 100;
L2_eff_lpc = sqrt(mean(error_eff_lpc.^2)) / (mean(abs(xqua_lpc(:,2)))) * 100;

% Compute standard deviation percentage ranges
% 1-sigma
lower_bound_FC_1s_lpc = mean_FC_lpc - std_FC_lpc;
upper_bound_FC_1s_lpc = mean_FC_lpc + std_FC_lpc;

lower_bound_eff_1s_lpc = mean_eff_lpc - std_eff_lpc;
upper_bound_eff_1s_lpc = mean_eff_lpc + std_eff_lpc;

% 2-sigma
lower_bound_FC_2s_lpc = mean_FC_lpc - 2*std_FC_lpc;
upper_bound_FC_2s_lpc = mean_FC_lpc + 2*std_FC_lpc;

lower_bound_eff_2s_lpc = mean_eff_lpc - 2*std_eff_lpc;
upper_bound_eff_2s_lpc = mean_eff_lpc + 2*std_eff_lpc;

% 3-sigma
lower_bound_FC_3s_lpc = mean_FC_lpc - 3*std_FC_lpc;
upper_bound_FC_3s_lpc = mean_FC_lpc + 3*std_FC_lpc;

lower_bound_eff_3s_lpc = mean_eff_lpc - 3*std_eff_lpc;
upper_bound_eff_3s_lpc = mean_eff_lpc + 3*std_eff_lpc;

% Count elements within the range
% 1-sigma
count_in_range_FC_1s_lpc = sum(FC_lpc >= lower_bound_FC_1s_lpc & FC_lpc <= upper_bound_FC_1s_lpc);
count_in_range_eff_1s_lpc = sum(eff_lpc >= lower_bound_eff_1s_lpc & eff_lpc <= upper_bound_eff_1s_lpc);

% 2-sigma
count_in_range_FC_2s_lpc = sum(FC_lpc >= lower_bound_FC_2s_lpc & FC_lpc <= upper_bound_FC_2s_lpc);
count_in_range_eff_2s_lpc = sum(eff_lpc >= lower_bound_eff_2s_lpc & eff_lpc <= upper_bound_eff_2s_lpc);

% 3-sigma
count_in_range_FC_3s_lpc = sum(FC_lpc >= lower_bound_FC_3s_lpc & FC_lpc <= upper_bound_FC_3s_lpc);
count_in_range_eff_3s_lpc = sum(eff_lpc >= lower_bound_eff_3s_lpc & eff_lpc <= upper_bound_eff_3s_lpc);

% Total number of elements
total_count_FC_lpc = length(FC_lpc);
total_count_eff_lpc = length(eff_lpc);

% Compute the proportion
sigma1_FC_lpc = (count_in_range_FC_1s_lpc / total_count_FC_lpc)*100;
sigma2_FC_lpc = (count_in_range_FC_2s_lpc / total_count_FC_lpc)*100;
sigma3_FC_lpc = (count_in_range_FC_3s_lpc / total_count_FC_lpc)*100;

sigma1_eff_lpc = (count_in_range_eff_1s_lpc / total_count_eff_lpc)*100;
sigma2_eff_lpc = (count_in_range_eff_2s_lpc / total_count_eff_lpc)*100;
sigma3_eff_lpc = (count_in_range_eff_3s_lpc / total_count_eff_lpc)*100;

%min-max
min_eff_lpc = min(eff_lpc);
min_FC_lpc = min(FC_lpc);

max_eff_lpc = max(eff_lpc);
max_FC_lpc = max(FC_lpc);


% Create the table data with sigma values
data_lpc = {'Statistics', 'Flow Capacity', 'Efficiency'; 
        'Range', range_FC_lpc, range_eff_lpc; 
        'Minimum Value', min_FC_lpc, min_eff_lpc
        'Maximum Value', max_FC_lpc, max_eff_lpc
        'Mean', mean_FC_lpc, mean_eff_lpc; 
        'Standard Deviation', std_FC_lpc, std_eff_lpc;
        'Absolute Error (%)', L1_fc_lpc, L1_eff_lpc; 
        'RMS Error (%)', L2_fc_lpc, L2_eff_lpc;
        '1σ (%)', sigma1_FC_lpc, sigma1_eff_lpc; 
        '2σ (%)', sigma2_FC_lpc, sigma2_eff_lpc; 
        '3σ (%)', sigma3_FC_lpc, sigma3_eff_lpc};

% Define table dimensions
numRows = size(data_lpc, 1);
numCols = size(data_lpc, 2);

% Create a figure
figure;
hold on;

% Set the figure size and remove axes
set(gcf, 'Units', 'pixels', 'Position', [100, 100, 600, 400]); % Increased height for new rows
axis off;

% Define table parameters
cellWidth = 120; % Width of each cell
cellHeight = 30; % Height of each cell
startX = 50; % Starting X position
startY = 370; % Adjusted for extra rows

% Draw the table borders
for row = 0:numRows
    y = startY - row * cellHeight;
    line([startX, startX + numCols * cellWidth], [y, y], 'Color', 'k', 'LineWidth', 1);
end

for col = 0:numCols
    x = startX + col * cellWidth;
    line([x, x], [startY, startY - numRows * cellHeight], 'Color', 'k', 'LineWidth', 1);
end

% Fill in the table data
for row = 1:numRows
    for col = 1:numCols
        x = startX + (col - 0.5) * cellWidth;
        y = startY - (row - 0.5) * cellHeight;
        
        % Convert numeric values to strings and format percentages
        if isnumeric(data_lpc{row, col})
            textStr = sprintf('%.2f', data_lpc{row, col}); 
        else
            textStr = data_lpc{row, col}; % Keep string values as they are
        end
        
        text(x, y, textStr, 'FontSize', 10, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end
end

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'LPC_statistics.png'));
hold off;

combinedmat3_lpc = [numbered_column3_lpc, x3_lpc, y_qua_lpc];

%% %%%%%%% Quantification, HPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("• High Pressure Compressor (HPC)")
% Take the result of the isolation out
x3_hpc = iso_hpc(:, 2:14);
numbered_column3_hpc = iso_hpc(:,1);

y_qua_hpc = APPROX2_C(x3_hpc);

%% %%%%% COMPARISON AND FIGURE %%%%%%%%%%%%%
FC_hpc = y_qua_hpc(:,1);
eff_hpc = y_qua_hpc(:,2);


% Plot 1: HPC_comparison_FlowDeg
% Normalise the numbered column
numbered_column_xqua_hpc = (1:size(xqua_hpc, 1))' + numbered_column3_hpc(1);

figure;
plot(numbered_column3_hpc, (FC_hpc))
hold on
plot(numbered_column_xqua_hpc, (xqua_hpc(:,1)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-7.5 1]);
xlabel('Dataset')
ylabel('Flow Capacity (%)')
legend({'Output', 'Target'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
 saveas(gcf, fullfile('charts', 'HPC_comparison_FlowDeg.png'));

% Plot 2: HPC_comparison_FlowDeg
figure;
plot(numbered_column3_hpc, (eff_hpc))
hold on
plot(numbered_column_xqua_hpc, (xqua_hpc(:,2)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-7.5 1]);
xlabel('Dataset')
ylabel('Efficency (%)')
legend({'Output', 'Target'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Set font to Times New Roman
%set(gca, 'FontName', 'Times New Roman');

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'HPC_comparison_FlowEff.png'));

%%%%%% PERFORMANCE STATISTICS %%%%%

% Calculate the range, mean, and standard deviation for array A (FC)
range_FC_hpc = range(FC_hpc);
mean_FC_hpc = mean(FC_hpc);
std_FC_hpc = std(FC_hpc);

% Range Mean StdDev for array B (Eff)
range_eff_hpc = range(eff_hpc);
mean_eff_hpc = mean(eff_hpc);
std_eff_hpc = std(eff_hpc);

% Error
size(FC_hpc)
size(xqua_hpc)
if size(FC_hpc, 1) > size(xqua_hpc,1)
    FC_hpc = FC_hpc(1:size(xqua_hpc,1), :);
    eff_hpc = eff_hpc(1:size(xqua_hpc,1), :);
    
    error_fc_hpc = FC_hpc - xqua_hpc(:,1);
    error_eff_hpc = eff_hpc - xqua_hpc(:,1);

elseif size(FC_hpc, 1) < size(xqua_hpc,1)
    xqua_trunc_1_hpc = xqua_hpc(1:size(FC_hpc,1), 1);
    xqua_trunc_2_hpc = xqua_hpc(1:size(eff_hpc,1), 2);
    
    error_fc_hpc = FC_hpc - xqua_trunc_1_hpc;
    error_eff_hpc = eff_hpc - xqua_trunc_2_hpc;
else
    error_fc_hpc = FC_hpc - xqua_hpc(:,1);
    error_eff_hpc = eff_hpc - xqua_hpc(:,2);
end

% L1 Norm (Sum of absolute differences)
L1_fc_hpc = sum(abs(error_fc_hpc)) / sum(abs(FC_hpc)) * 100;
L1_eff_hpc = sum(abs(error_eff_hpc)) / sum(abs(eff_hpc)) * 100;

% L2 Norm (Square root of sum of squared differences)
L2_fc_hpc = sqrt(mean(error_fc_hpc.^2)) / (mean(abs(xqua_hpc(:,2)))) * 100;
L2_eff_hpc = sqrt(mean(error_eff_hpc.^2)) / (mean(abs(xqua_hpc(:,2)))) * 100;

% Compute standard deviation percentage ranges
% 1-sigma
lower_bound_FC_1s_hpc = mean_FC_hpc - std_FC_hpc;
upper_bound_FC_1s_hpc = mean_FC_hpc + std_FC_hpc;

lower_bound_eff_1s_hpc = mean_eff_hpc - std_eff_hpc;
upper_bound_eff_1s_hpc = mean_eff_hpc + std_eff_hpc;

% 2-sigma
lower_bound_FC_2s_hpc = mean_FC_hpc - 2*std_FC_hpc;
upper_bound_FC_2s_hpc = mean_FC_hpc + 2*std_FC_hpc;

lower_bound_eff_2s_hpc = mean_eff_hpc - 2*std_eff_hpc;
upper_bound_eff_2s_hpc = mean_eff_hpc + 2*std_eff_hpc;

% 3-sigma
lower_bound_FC_3s_hpc = mean_FC_hpc - 3*std_FC_hpc;
upper_bound_FC_3s_hpc = mean_FC_hpc + 3*std_FC_hpc;

lower_bound_eff_3s_hpc = mean_eff_hpc - 3*std_eff_hpc;
upper_bound_eff_3s_hpc = mean_eff_hpc + 3*std_eff_hpc;

% Count elements within the range
% 1-sigma
count_in_range_FC_1s_hpc = sum(FC_hpc >= lower_bound_FC_1s_hpc & FC_hpc <= upper_bound_FC_1s_hpc);
count_in_range_eff_1s_hpc = sum(eff_hpc >= lower_bound_eff_1s_hpc & eff_hpc <= upper_bound_eff_1s_hpc);

% 2-sigma
count_in_range_FC_2s_hpc = sum(FC_hpc >= lower_bound_FC_2s_hpc & FC_hpc <= upper_bound_FC_2s_hpc);
count_in_range_eff_2s_hpc = sum(eff_hpc >= lower_bound_eff_2s_hpc & eff_hpc <= upper_bound_eff_2s_hpc);

% 3-sigma
count_in_range_FC_3s_hpc = sum(FC_hpc >= lower_bound_FC_3s_hpc & FC_hpc <= upper_bound_FC_3s_hpc);
count_in_range_eff_3s_hpc = sum(eff_hpc >= lower_bound_eff_3s_hpc & eff_hpc <= upper_bound_eff_3s_hpc);

% Total number of elements
total_count_FC_hpc = length(FC_hpc);
total_count_eff_hpc = length(eff_hpc);

% Compute the proportion
sigma1_FC_hpc = (count_in_range_FC_1s_hpc / total_count_FC_hpc)*100;
sigma2_FC_hpc = (count_in_range_FC_2s_hpc / total_count_FC_hpc)*100;
sigma3_FC_hpc = (count_in_range_FC_3s_hpc / total_count_FC_hpc)*100;

sigma1_eff_hpc = (count_in_range_eff_1s_hpc / total_count_eff_hpc)*100;
sigma2_eff_hpc = (count_in_range_eff_2s_hpc / total_count_eff_hpc)*100;
sigma3_eff_hpc = (count_in_range_eff_3s_hpc / total_count_eff_hpc)*100;

%min-max
min_eff_hpc = min(eff_hpc);
min_FC_hpc = min(FC_hpc);

max_eff_hpc = max(eff_hpc);
max_FC_hpc = max(FC_hpc);


% Create the table data with sigma values
data_hpc = {'Statistics', 'Flow Capacity', 'Efficiency'; 
        'Range', range_FC_hpc, range_eff_hpc; 
        'Minimum Value', min_FC_hpc, min_eff_hpc
        'Maximum Value', max_FC_hpc, max_eff_hpc
        'Mean', mean_FC_hpc, mean_eff_hpc; 
        'Standard Deviation', std_FC_hpc, std_eff_hpc;
        'Absolute Error (%)', L1_fc_hpc, L1_eff_hpc; 
        'RMS Error (%)', L2_fc_hpc, L2_eff_hpc;
        '1σ (%)', sigma1_FC_hpc, sigma1_eff_hpc; 
        '2σ (%)', sigma2_FC_hpc, sigma2_eff_hpc; 
        '3σ (%)', sigma3_FC_hpc, sigma3_eff_hpc};

% Define table dimensions
numRows = size(data_hpc, 1);
numCols = size(data_hpc, 2);

% Create a figure
figure;
hold on;

% Set the figure size and remove axes
set(gcf, 'Units', 'pixels', 'Position', [100, 100, 600, 400]); % Increased height for new rows
axis off;

% Define table parameters
cellWidth = 120; % Width of each cell
cellHeight = 30; % Height of each cell
startX = 50; % Starting X position
startY = 370; % Adjusted for extra rows

% Draw the table borders
for row = 0:numRows
    y = startY - row * cellHeight;
    line([startX, startX + numCols * cellWidth], [y, y], 'Color', 'k', 'LineWidth', 1);
end

for col = 0:numCols
    x = startX + col * cellWidth;
    line([x, x], [startY, startY - numRows * cellHeight], 'Color', 'k', 'LineWidth', 1);
end

% Fill in the table data
for row = 1:numRows
    for col = 1:numCols
        x = startX + (col - 0.5) * cellWidth;
        y = startY - (row - 0.5) * cellHeight;
        
        % Convert numeric values to strings and format percentages
        if isnumeric(data_hpc{row, col})
            textStr = sprintf('%.2f', data_hpc{row, col}); 
        else
            textStr = data_hpc{row, col}; % Keep string values as they are
        end
        
        text(x, y, textStr, 'FontSize', 10, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end
end

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'HPC_statistics.png'));
hold off;

combinedmat3_hpc = [numbered_column3_hpc, x3_hpc, y_qua_hpc];

%% %%%%%%% Quantification, HPT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
disp("• High Pressure Turbine (HPT)")


% Take the result of the isolation out
x3_hpt = iso_hpt(:, 2:14);
numbered_column3_hpt = iso_hpt(:,1);

%Neural Network Commences
y_qua_hpt = APPROX3_C(x3_hpt);

%% %%%%% COMPARISON AND FIGURE %%%%%%%%%%%%%
FC_hpt = y_qua_hpt(:,1);
eff_hpt = y_qua_hpt(:,2);


% Plot 1: HPC_comparison_FlowDeg
% Normalise the numbered column
numbered_column_xqua_hpt = (1:size(xqua_hpt, 1))' + numbered_column3_hpt(1);

figure;
plot(numbered_column3_hpt, (FC_hpt))
hold on
plot(numbered_column_xqua_hpt, (xqua_hpt(:,1)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-1 7]);
xlabel('Dataset')
ylabel('Flow Capacity (%)')
legend({'Output', 'Target'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
 saveas(gcf, fullfile('charts', 'HPT_comparison_FlowDeg.png'));

% Plot 2: HPC_comparison_Floweff
figure;
plot(numbered_column3_hpt, (eff_hpt))
hold on
plot(numbered_column_xqua_hpt, (xqua_hpt(:,2)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-7.5 1]);
xlabel('Dataset')
ylabel('Efficency (%)')
legend({'Output', 'Target'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Set font to Times New Roman
%set(gca, 'FontName', 'Times New Roman');

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'HPT_comparison_FlowEff.png'));

%%%%%% PERFORMANCE STATISTICS %%%%%

% Calculate the range, mean, and standard deviation for array A (FC)
range_FC_hpt = range(FC_hpt);
mean_FC_hpt = mean(FC_hpt);
std_FC_hpt = std(FC_hpt);

% Range Mean StdDev for array B (Eff)
range_eff_hpt = range(eff_hpt);
mean_eff_hpt = mean(eff_hpt);
std_eff_hpt = std(eff_hpt);

% Error
if size(FC_hpt, 1) > size(xqua_hpt,1)
    FC_hpt = FC_hpt(1:size(xqua_hpt,1), :);
    eff_hpt = eff_hpt(1:size(xqua_hpt,1), :);
    
    error_fc_hpt = FC_hpt - xqua_hpt(:,1);
    error_eff_hpt = eff_hpt - xqua_hpt(:,1);

elseif size(FC_hpt, 1) < size(xqua_hpt,1)
    xqua_trunc_1_hpt = xqua_hpt(1:size(FC_hpt,1), 1);
    xqua_trunc_2_hpt = xqua_hpt(1:size(eff_hpt,1), 2);
    
    error_fc_hpt = FC_hpt - xqua_trunc_1_hpt;
    error_eff_hpt = eff_hpt - xqua_trunc_2_hpt;
else
    error_fc_hpt = FC_hpt - xqua_hpt;
    error_eff_hpt = eff_hpt - xqua_hpt;
end

% L1 Norm (Sum of absolute differences)
L1_fc_hpt = sum(abs(error_fc_hpt)) / sum(abs(FC_hpt)) * 100;
L1_eff_hpt = sum(abs(error_eff_hpt)) / sum(abs(eff_hpt)) * 100;

% L2 Norm (Square root of sum of squared differences)
L2_fc_hpt = sqrt(mean(error_fc_hpt.^2)) / (mean(abs(xqua_hpt(:,2)))) * 100;
L2_eff_hpt = sqrt(mean(error_eff_hpt.^2)) / (mean(abs(xqua_hpt(:,2)))) * 100;

% Compute standard deviation percentage ranges
% 1-sigma
lower_bound_FC_1s_hpt = mean_FC_hpt - std_FC_hpt;
upper_bound_FC_1s_hpt = mean_FC_hpt + std_FC_hpt;

lower_bound_eff_1s_hpt = mean_eff_hpt - std_eff_hpt;
upper_bound_eff_1s_hpt = mean_eff_hpt + std_eff_hpt;

% 2-sigma
lower_bound_FC_2s_hpt = mean_FC_hpt - 2*std_FC_hpt;
upper_bound_FC_2s_hpt = mean_FC_hpt + 2*std_FC_hpt;

lower_bound_eff_2s_hpt = mean_eff_hpt - 2*std_eff_hpt;
upper_bound_eff_2s_hpt = mean_eff_hpt + 2*std_eff_hpt;

% 3-sigma
lower_bound_FC_3s_hpt = mean_FC_hpt - 3*std_FC_hpt;
upper_bound_FC_3s_hpt = mean_FC_hpt + 3*std_FC_hpt;

lower_bound_eff_3s_hpt = mean_eff_hpt - 3*std_eff_hpt;
upper_bound_eff_3s_hpt = mean_eff_hpt + 3*std_eff_hpt;

% Count elements within the range
% 1-sigma
count_in_range_FC_1s_hpt = sum(FC_hpt >= lower_bound_FC_1s_hpt & FC_hpt <= upper_bound_FC_1s_hpt);
count_in_range_eff_1s_hpt = sum(eff_hpt >= lower_bound_eff_1s_hpt & eff_hpt <= upper_bound_eff_1s_hpt);

% 2-sigma
count_in_range_FC_2s_hpt = sum(FC_hpt >= lower_bound_FC_2s_hpt & FC_hpt <= upper_bound_FC_2s_hpt);
count_in_range_eff_2s_hpt = sum(eff_hpt >= lower_bound_eff_2s_hpt & eff_hpt <= upper_bound_eff_2s_hpt);

% 3-sigma
count_in_range_FC_3s_hpt = sum(FC_hpt >= lower_bound_FC_3s_hpt & FC_hpt <= upper_bound_FC_3s_hpt);
count_in_range_eff_3s_hpt = sum(eff_hpt >= lower_bound_eff_3s_hpt & eff_hpt <= upper_bound_eff_3s_hpt);

% Total number of elements
total_count_FC_hpt = length(FC_hpt);
total_count_eff_hpt = length(eff_hpt);

% Compute the proportion
sigma1_FC_hpt = (count_in_range_FC_1s_hpt / total_count_FC_hpt)*100;
sigma2_FC_hpt = (count_in_range_FC_2s_hpt / total_count_FC_hpt)*100;
sigma3_FC_hpt = (count_in_range_FC_3s_hpt / total_count_FC_hpt)*100;

sigma1_eff_hpt = (count_in_range_eff_1s_hpt / total_count_eff_hpt)*100;
sigma2_eff_hpt = (count_in_range_eff_2s_hpt / total_count_eff_hpt)*100;
sigma3_eff_hpt = (count_in_range_eff_3s_hpt / total_count_eff_hpt)*100;

%min-max
min_eff_hpt = min(eff_hpt);
min_FC_hpt = min(FC_hpt);

max_eff_hpt = max(eff_hpt);
max_FC_hpt = max(FC_hpt);


% Create the table data with sigma values
data_hpt = {'Statistics', 'Flow Capacity', 'Efficiency'; 
        'Range', range_FC_hpt, range_eff_hpt; 
        'Minimum Value', min_FC_hpt, min_eff_hpt
        'Maximum Value', max_FC_hpt, max_eff_hpt
        'Mean', mean_FC_hpt, mean_eff_hpt; 
        'Standard Deviation', std_FC_hpt, std_eff_hpt;
        'Absolute Error (%)', L1_fc_hpt, L1_eff_hpt; 
        'RMS Error (%)', L2_fc_hpt, L2_eff_hpt;
        '1σ (%)', sigma1_FC_hpt, sigma1_eff_hpt; 
        '2σ (%)', sigma2_FC_hpt, sigma2_eff_hpt; 
        '3σ (%)', sigma3_FC_hpt, sigma3_eff_hpt};

% Define table dimensions
numRows = size(data_hpt, 1);
numCols = size(data_hpt, 2);

% Create a figure
figure;
hold on;

% Set the figure size and remove axes
set(gcf, 'Units', 'pixels', 'Position', [100, 100, 600, 400]); % Increased height for new rows
axis off;

% Define table parameters
cellWidth = 120; % Width of each cell
cellHeight = 30; % Height of each cell
startX = 50; % Starting X position
startY = 370; % Adjusted for extra rows

% Draw the table borders
for row = 0:numRows
    y = startY - row * cellHeight;
    line([startX, startX + numCols * cellWidth], [y, y], 'Color', 'k', 'LineWidth', 1);
end

for col = 0:numCols
    x = startX + col * cellWidth;
    line([x, x], [startY, startY - numRows * cellHeight], 'Color', 'k', 'LineWidth', 1);
end

% Fill in the table data
for row = 1:numRows
    for col = 1:numCols
        x = startX + (col - 0.5) * cellWidth;
        y = startY - (row - 0.5) * cellHeight;
        
        % Convert numeric values to strings and format percentages
        if isnumeric(data_hpt{row, col})
            textStr = sprintf('%.2f', data_hpt{row, col}); 
        else
            textStr = data_hpt{row, col}; % Keep string values as they are
        end
        
        text(x, y, textStr, 'FontSize', 10, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end
end

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'HPT_statistics.png'));
hold off;

combinedmat3_hpt = [numbered_column3_hpt, x3_hpt, y_qua_hpt];

%% %%%%%%% Quantification, LPT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("• Low Pressure Turbine (LPT)")

% Take the result of the isolation out
x3_lpt = iso_lpt(:, 2:14);
numbered_column3_lpt = iso_lpt(:,1);

%Neural Network Commences
y_qua_lpt = APPROX4_C(x3_lpt);

%% %%%%% COMPARISON AND FIGURE %%%%%%%%%%%%%
FC_lpt = y_qua_lpt(:,1);
eff_lpt = y_qua_lpt(:,2);


% Plot 1: HPC_comparison_FlowDeg
% Normalise the numbered column
numbered_column_xqua_lpt = (1:size(xqua_lpt, 1))' + numbered_column3_lpt(end) - length(xqua_lpt);

figure;
plot(numbered_column3_lpt, (FC_lpt))
hold on
plot(numbered_column_xqua_lpt, (xqua_lpt(:,1)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-1 7]);
xlabel('Dataset')
ylabel('Flow Capacity (%)')
legend({'Target', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
 saveas(gcf, fullfile('charts', 'LPT_comparison_FlowDeg.png'));

% Plot 2: HPC_comparison_FlowDeg
figure;
plot(numbered_column3_lpt, (eff_lpt))
hold on
plot(numbered_column_xqua_lpt, (xqua_lpt(:,2)))
% Subplot parameters
xlim([0 length(xiso)]); 
ylim([-7.5 1]);
xlabel('Dataset')
ylabel('Efficency (%)')
legend({'Target', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
hold off

% Set font to Times New Roman
%set(gca, 'FontName', 'Times New Roman');

% Change the size of the plot to a 1:3 ratio landscape
set(gcf, 'Position', [100, 100, 1000, 400]); % [left, bottom, width, height]

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'LPT_comparison_FlowEff.png'));

%%%%%% PERFORMANCE STATISTICS %%%%%

% Calculate the range, mean, and standard deviation for array A (FC)
range_FC_lpt = range(FC_lpt);
mean_FC_lpt = mean(FC_lpt);
std_FC_lpt = std(FC_lpt);

% Range Mean StdDev for array B (Eff)
range_eff_lpt = range(eff_lpt);
mean_eff_lpt = mean(eff_lpt);
std_eff_lpt = std(eff_lpt);

% Error
if size(FC_lpt, 1) > size(xqua_lpt,1)
    FC_lpt = FC_lpt(1:size(xqua_lpt,1), :);
    eff_lpt = eff_lpt(1:size(xqua_lpt,1), :);
    
    error_fc_lpt = FC_lpt - xqua_lpt(:,1);
    error_eff_lpt = eff_lpt - xqua_lpt(:,2);

elseif size(FC_lpt, 1) < size(xqua_lpt,1)
    xqua_trunc_1_lpt = xqua_lpt(1:size(FC_lpt,1), 1);
    xqua_trunc_2_lpt = xqua_lpt(1:size(eff_lpt,1), 2);
    
    error_fc_lpt = FC_lpt - xqua_trunc_1_lpt;
    error_eff_lpt = eff_lpt - xqua_trunc_2_lpt;
else
    error_fc_lpt = FC_lpt - xqua_lpt(:,1);
    error_eff_lpt = eff_lpt - xqua_lpt(:,1);
end

% L1 Norm (Sum of absolute differences)
L1_fc_lpt = sum(abs(error_fc_lpt)) / sum(abs(FC_lpt)) * 100;
L1_eff_lpt = sum(abs(error_eff_lpt)) / sum(abs(eff_lpt)) * 100;

% L2 Norm (Square root of sum of squared differences)
L2_fc_lpt = sqrt(mean(error_fc_lpt.^2)) / (mean(abs(xqua_lpt(:,1)))) * 100;
L2_eff_lpt = sqrt(mean(error_eff_lpt.^2)) / (mean(abs(xqua_lpt(:,2)))) * 100;

% Compute standard deviation percentage ranges
% 1-sigma
lower_bound_FC_1s_lpt = mean_FC_lpt - std_FC_lpt;
upper_bound_FC_1s_lpt = mean_FC_lpt + std_FC_lpt;

lower_bound_eff_1s_lpt = mean_eff_lpt - std_eff_lpt;
upper_bound_eff_1s_lpt = mean_eff_lpt + std_eff_lpt;

% 2-sigma
lower_bound_FC_2s_lpt = mean_FC_lpt - 2*std_FC_lpt;
upper_bound_FC_2s_lpt = mean_FC_lpt + 2*std_FC_lpt;

lower_bound_eff_2s_lpt = mean_eff_lpt - 2*std_eff_lpt;
upper_bound_eff_2s_lpt = mean_eff_lpt + 2*std_eff_lpt;

% 3-sigma
lower_bound_FC_3s_lpt = mean_FC_lpt - 3*std_FC_lpt;
upper_bound_FC_3s_lpt = mean_FC_lpt + 3*std_FC_lpt;

lower_bound_eff_3s_lpt = mean_eff_lpt - 3*std_eff_lpt;
upper_bound_eff_3s_lpt = mean_eff_lpt + 3*std_eff_lpt;

% Count elements within the range
% 1-sigma
count_in_range_FC_1s_lpt = sum(FC_lpt >= lower_bound_FC_1s_lpt & FC_lpt <= upper_bound_FC_1s_lpt);
count_in_range_eff_1s_lpt = sum(eff_lpt >= lower_bound_eff_1s_lpt & eff_lpt <= upper_bound_eff_1s_lpt);

% 2-sigma
count_in_range_FC_2s_lpt = sum(FC_lpt >= lower_bound_FC_2s_lpt & FC_lpt <= upper_bound_FC_2s_lpt);
count_in_range_eff_2s_lpt = sum(eff_lpt >= lower_bound_eff_2s_lpt & eff_lpt <= upper_bound_eff_2s_lpt);

% 3-sigma
count_in_range_FC_3s_lpt = sum(FC_lpt >= lower_bound_FC_3s_lpt & FC_lpt <= upper_bound_FC_3s_lpt);
count_in_range_eff_3s_lpt = sum(eff_lpt >= lower_bound_eff_3s_lpt & eff_lpt <= upper_bound_eff_3s_lpt);

% Total number of elements
total_count_FC_lpt = length(FC_lpt);
total_count_eff_lpt = length(eff_lpt);

% Compute the proportion
sigma1_FC_lpt = (count_in_range_FC_1s_lpt / total_count_FC_lpt)*100;
sigma2_FC_lpt = (count_in_range_FC_2s_lpt / total_count_FC_lpt)*100;
sigma3_FC_lpt = (count_in_range_FC_3s_lpt / total_count_FC_lpt)*100;

sigma1_eff_lpt = (count_in_range_eff_1s_lpt / total_count_eff_lpt)*100;
sigma2_eff_lpt = (count_in_range_eff_2s_lpt / total_count_eff_lpt)*100;
sigma3_eff_lpt = (count_in_range_eff_3s_lpt / total_count_eff_lpt)*100;

%min-max
min_eff_lpt = min(eff_lpt);
min_FC_lpt = min(FC_lpt);

max_eff_lpt = max(eff_lpt);
max_FC_lpt = max(FC_lpt);


% Create the table data with sigma values
data_lpt = {'Statistics', 'Flow Capacity', 'Efficiency'; 
        'Range', range_FC_lpt, range_eff_lpt; 
        'Minimum Value', min_FC_lpt, min_eff_lpt
        'Maximum Value', max_FC_lpt, max_eff_lpt
        'Mean', mean_FC_lpt, mean_eff_lpt; 
        'Standard Deviation', std_FC_lpt, std_eff_lpt;
        'Absolute Error (%)', L1_fc_lpt, L1_eff_lpt; 
        'RMS Error (%)', L2_fc_lpt, L2_eff_lpt;
        '1σ (%)', sigma1_FC_lpt, sigma1_eff_lpt; 
        '2σ (%)', sigma2_FC_lpt, sigma2_eff_lpt; 
        '3σ (%)', sigma3_FC_lpt, sigma3_eff_lpt};

% Define table dimensions
numRows = size(data_lpt, 1);
numCols = size(data_lpt, 2);

% Create a figure
figure;
hold on;

% Set the figure size and remove axes
set(gcf, 'Units', 'pixels', 'Position', [100, 100, 600, 400]); % Increased height for new rows
axis off;

% Define table parameters
cellWidth = 120; % Width of each cell
cellHeight = 30; % Height of each cell
startX = 50; % Starting X position
startY = 370; % Adjusted for extra rows

% Draw the table borders
for row = 0:numRows
    y = startY - row * cellHeight;
    line([startX, startX + numCols * cellWidth], [y, y], 'Color', 'k', 'LineWidth', 1);
end

for col = 0:numCols
    x = startX + col * cellWidth;
    line([x, x], [startY, startY - numRows * cellHeight], 'Color', 'k', 'LineWidth', 1);
end

% Fill in the table data
for row = 1:numRows
    for col = 1:numCols
        x = startX + (col - 0.5) * cellWidth;
        y = startY - (row - 0.5) * cellHeight;
        
        % Convert numeric values to strings and format percentages
        if isnumeric(data_lpt{row, col})
            textStr = sprintf('%.2f', data_lpt{row, col}); 
        else
            textStr = data_lpt{row, col}; % Keep string values as they are
        end
        
        text(x, y, textStr, 'FontSize', 10, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end
end

% Save the figure as a PNG file
saveas(gcf, fullfile('charts', 'LPT_statistics.png'));
hold off;


combinedmat3_lpt = [numbered_column3_lpt, x3_lpt, y_qua_lpt];

%% %%%%%%% Excel Output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% disp("Writing result to Excel...")
% 
% % Headers for the sheets (converted to cell array)
% headers_A = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN'};
% headers_B = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN', 'Detection Result'};
% headers_C = {'Case-No', 'P Total 3', 'T Total 3', 'P Total 5', 'T Total 5', 'P Total 6', 'T Total 6', 'P Total 10', 'T Total 10', 'P Total 12', 'T Total 12', 'Fuel Flow 0', 'Comp 1 PCN', 'Comp 3 PCN', 'Flow Degradation (%)', 'Flow Efficiency(%)'};
% 
% %Combining x1 with case number
% A = [numbered_column, x1]; 
% 
% % Convert numeric data in A to cell array
% data_A = [headers_A; num2cell(A)];
% data_B = [headers_B; num2cell(combinedfiltered)];
% data_C = [headers_C; num2cell(combinedmat3_lpc)];
% data_D = [headers_C; num2cell(combinedmat3_hpc)];
% data_E = [headers_C; num2cell(combinedmat3_hpt)];
% data_F = [headers_C; num2cell(combinedmat3_lpt)];
% 
% % Define the filename
% filename = 'NN-Results_qua-test.xlsx';
% 
% % Write data_A to the first sheet with headers
% writecell(data_A, filename, 'Sheet', 'Input_Case');
% writecell(data_B, filename, 'Sheet', 'Detection');
% writecell(data_C, filename, 'Sheet', 'Iso-Qua_LPC');
% writecell(data_D, filename, 'Sheet', 'Iso-Qua_HPC');
% writecell(data_E, filename, 'Sheet', 'Iso-Qua_HPT');
% writecell(data_F, filename, 'Sheet', 'Iso-Qua_LPT');
% 
% disp("Excel Generation Finished.")
%%
disp("RUN FINISHED!")