% KITTI OBJECT DETECTION AND ORIENTATION ESTIMATION BENCHMARK STATISTICS
clear all; close all; clc;
disp('======= KITTI DevKit Statistics =======');
disp('Computing statistics of Cars, Pedestrians and Cyclists in training set.');
disp('Please wait ...');

% options
root_dir = 'C:\EPFL_tools\KITTI';

% get label directory and number of images
label_dir = fullfile(root_dir,'predicted\universe\data');
% get the images labeled for this testing
nimages = length(dir(fullfile(label_dir, '*.txt')));

folder = label_dir; % replace with the path to your folder
file_list = dir(fullfile(folder, '*.txt')); % get only the .txt files
file_list = file_list(~[file_list.isdir]); % exclude directories
file_names = {file_list.name}; % extract the file names
% Trim leading zeros and remove .txt extension
for i = 1:length(file_names)
    % Get the current file name
    current_name = file_names{i};
    % Trim leading zeros
    while current_name(1) == '0'
        current_name(1) = [];
    end
    
    % Remove the .txt extension
    current_name = current_name(1:end-4);
       
    % add the file "zero" 
    if strcmp(file_names{i},'000000.txt')
        current_name = '0';
    end
    % Replace the old name with the new name
    file_names{i} = current_name;
end

% init statistics
cars.level = zeros(1,4);
cars.level = zeros(1,4);
peds.level = zeros(1,4);
cycs.level = zeros(1,4);
vans.level = zeros(1,4);
trucks.level = zeros(1,4);
pers_sitting.level = zeros(1,4);
trams.level = zeros(1,4);

% compute statistics
for j=1:nimages
  objects = readLabels(label_dir,str2double(file_names{j}));
  for k=1:length(objects)
    if strcmp(objects(k).type,'Car')
      cars.level(objects(k).level) = cars.level(objects(k).level) + 1;
    end
    if strcmp(objects(k).type,'Pedestrian')
      peds.level(objects(k).level)  = peds.level(objects(k).level)  + 1;
    end
    if strcmp(objects(k).type,'Cyclist')
      cycs.level(objects(k).level)  = cycs.level(objects(k).level)  + 1;
    end
    if strcmp(objects(k).type,'Van')
      vans.level(objects(k).level)  = vans.level(objects(k).level)  + 1;
    end
    if strcmp(objects(k).type,'Truck')
      trucks.level(objects(k).level)  = trucks.level(objects(k).level)  + 1;
    end
    if strcmp(objects(k).type,'Tram')
      trams.level(objects(k).level)  = trams.level(objects(k).level)  + 1;
    end
    if strcmp(objects(k).type,'Person_sitting')
      pers_sitting.level(objects(k).level)  = pers_sitting.level(objects(k).level)  + 1;
    end
  end    
end

% plot statistics
fprintf('Cars: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',cars.level);
fprintf('Pedestrians: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',peds.level);
fprintf('Cyclists: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',cycs.level);
fprintf('Vans: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',vans.level);
fprintf('Trucks: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',trucks.level);
fprintf('Trams: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',trams.level);
fprintf('Person sitting: Easy: %d, moderate: %d, hard: %d, unknown: %d\n',pers_sitting.level);