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

% POST PROCESSING
% Generate Figures
generate_comparison_figures1(y_qua_lpc, xqua_lpc, numbered_column3_lpc, xiso, "LPC")

combinedmat3_lpc = [numbered_column3_lpc, x3_lpc, y_qua_lpc];

%% %%%%%%% Quantification, HPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("• High Pressure Compressor (HPC)")
% Take the result of the isolation out
x3_hpc = iso_hpc(:, 2:14);
numbered_column3_hpc = iso_hpc(:,1);

y_qua_hpc = APPROX2_C(x3_hpc);

generate_comparison_figures1(y_qua_hpc, xqua_hpc, numbered_column3_hpc, xiso, "HPC")

combinedmat3_hpc = [numbered_column3_hpc, x3_hpc, y_qua_hpc];

%% %%%%%%% Quantification, HPT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
disp("• High Pressure Turbine (HPT)")


% Take the result of the isolation out
x3_hpt = iso_hpt(:, 2:14);
numbered_column3_hpt = iso_hpt(:,1);

%Neural Network Commences
y_qua_hpt = APPROX3_C(x3_hpt);

generate_comparison_figures1(y_qua_hpt, xqua_hpt, numbered_column3_hpt, xiso, "HPT")

combinedmat3_hpt = [numbered_column3_hpt, x3_hpt, y_qua_hpt];

%% %%%%%%% Quantification, LPT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("• Low Pressure Turbine (LPT)")

% Take the result of the isolation out
x3_lpt = iso_lpt(:, 2:14);
numbered_column3_lpt = iso_lpt(:,1);

%Neural Network Commences
y_qua_lpt = APPROX4_C(x3_lpt);
generate_comparison_figures2(y_qua_lpt, xqua_lpt, numbered_column3_lpt, xiso, "LPT")

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
disp("RUN FINISHED!")

%% %%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FC, eff] = quares(y_qua)
    FC = y_qua(:,1);
    eff = y_qua(:,2);
end

function generate_comparison_figures1(y_qua, xqua, numbered_column, xiso, name_prefix)
    [FC, eff] = quares(y_qua);
    numbered_column_xqua = (1:size(xqua, 1))' + numbered_column(1);
    
    % Plot 1: Flow Capacity Comparison
    figure;
    plot(numbered_column, FC)
    hold on
    plot(numbered_column_xqua, xqua(:,1))
    xlim([0 length(xiso)]); 
    %ylim([-7.5 1]);
    xlabel('Dataset')
    ylabel('Flow Capacity (%)')
    legend({'Target', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
    hold off
    set(gcf, 'Position', [100, 100, 1000, 400]);
    saveas(gcf, fullfile('charts', strcat(name_prefix, '_comparison_FlowCap.png')));
    
    % Plot 2: Efficiency Comparison
    figure;
    plot(numbered_column, eff)
    hold on
    plot(numbered_column_xqua, xqua(:,2))
    xlim([0 length(xiso)]); 
    ylim([-7.5 1]);
    xlabel('Dataset')
    ylabel('Efficiency (%)')
    legend({'Target', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
    hold off
    set(gcf, 'Position', [100, 100, 1000, 400]);
    saveas(gcf, fullfile('charts', strcat(name_prefix, '_comparison_Eff.png')));
    
    % Compute Statistics
    stats = compute_statistics(FC, eff, xqua);
    
    % Save Statistics Table
    save_statistics_table(stats, name_prefix);
end

function generate_comparison_figures2(y_qua, xqua, numbered_column, xiso, name_prefix)
    [FC, eff] = quares(y_qua);

    numbered_column_xqua = (1:size(xqua, 1))' + numbered_column(length(numbered_column)) - length(xqua);
    
    % Plot 1: Flow Capacity Comparison
    figure;
    plot(numbered_column, FC)
    hold on
    plot(numbered_column_xqua, xqua(:,1))
    xlim([0 length(xiso)]); 
    ylim([-1 7]);
    xlabel('Dataset')
    ylabel('Flow Capacity (%)')
    legend({'Target', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
    hold off
    set(gcf, 'Position', [100, 100, 1000, 400]);
    saveas(gcf, fullfile('charts', strcat(name_prefix, '_comparison_FlowCap.png')));
    
    % Plot 2: Efficiency Comparison
    figure;
    plot(numbered_column, eff)
    hold on
    plot(numbered_column_xqua, xqua(:,2))
    xlim([0 length(xiso)]); 
    ylim([-7.5 1]);
    xlabel('Dataset')
    ylabel('Efficiency (%)')
    legend({'Target', 'Actual'}, 'Location', 'south', 'Box', 'off', 'Orientation', 'horizontal');
    hold off
    set(gcf, 'Position', [100, 100, 1000, 400]);
    saveas(gcf, fullfile('charts', strcat(name_prefix, '_comparison_Eff.png')));
    
    % Compute Statistics
    stats = compute_statistics(FC, eff, xqua);
    
    % Save Statistics Table
    save_statistics_table(stats, name_prefix);
end

function stats = compute_statistics(FC, eff, xqua)
    stats.range_FC = range(FC);
    stats.mean_FC = mean(FC);
    stats.std_FC = std(FC);
    stats.range_eff = range(eff);
    stats.mean_eff = mean(eff);
    stats.std_eff = std(eff);
    
    % Error computation
    if size(FC, 1) > size(xqua,1)
        FC = FC(1:size(xqua,1), :);
        eff = eff(1:size(xqua,1), :);
    elseif size(FC, 1) < size(xqua,1)
        xqua = xqua(1:size(FC,1), :);
    end
    
    error_fc = FC - xqua(:,1);
    error_eff = eff - xqua(:,2);
    
    % Error metrics
    stats.L1_fc = sum(abs(error_fc)) / sum(abs(FC)) * 100;
    stats.L1_eff = sum(abs(error_eff)) / sum(abs(eff)) * 100;
    stats.L2_fc = sqrt(mean(error_fc.^2)) / mean(abs(xqua(:,1))) * 100;
    stats.L2_eff = sqrt(mean(error_eff.^2)) / mean(abs(xqua(:,2))) * 100;
    
    % Min/Max Values
    stats.min_FC = min(FC);
    stats.max_FC = max(FC);
    stats.min_eff = min(eff);
    stats.max_eff = max(eff);
end

function save_statistics_table(stats, name_prefix)
    data = {'Statistics', 'Flow Capacity', 'Efficiency'; 
            'Range', stats.range_FC, stats.range_eff; 
            'Minimum', stats.min_FC, stats.min_eff;
            'Maximum', stats.max_FC, stats.max_eff;
            'Mean', stats.mean_FC, stats.mean_eff; 
            'Std Dev', stats.std_FC, stats.std_eff;
            'L1 Error (%)', stats.L1_fc, stats.L1_eff; 
            'L2 Error (%)', stats.L2_fc, stats.L2_eff};
    
    % Create figure
    % Define table dimensions
    numRows = size(data, 1);
    numCols = size(data, 2);
    
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
            if isnumeric(data{row, col})
                textStr = sprintf('%.2f', data{row, col}); 
            else
                textStr = data{row, col}; % Keep string values as they are
            end
            
            text(x, y, textStr, 'FontSize', 10, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
        end
    end
    
    % Save the figure as a PNG file
    saveas(gcf, fullfile('charts', 'LPT_statistics.png'));
    hold off;
    
    % Save figure
    saveas(gcf, fullfile('charts', strcat(name_prefix, '_statistics.png')));
end


