% KITTI OBJECT DETECTION AND ORIENTATION ESTIMATION BENCHMARK DEMONSTRATION
%
% This tool displays the images and the object labels for the benchmark and
% provides an entry point for writing your own interface to the data set.
% Before running this tool, set root_dir to the directory where you have
% downloaded the dataset. 'root_dir' must contain the subdirectory
% 'training', which in turn contains 'image_2', 'label_2' and 'calib'.
% For more information about the data format, please look into readme.txt.
%
% Usage:
%   → :   next frame
%   ← :   previous frame
%   ↑ :  +100 frames
%   ↓ :  -100 frames
%   m :  +1000 frames
%   n :  -1000 frames
%   q:    quit
%


% clear and close everything
clear all; close all;
disp('======= KITTI DevKit Demo =======');

data_set = 'training';

% get sub-directories
% cam = 2; % 2 = left color camera
% image_dir = fullfile(['C:\EPFL_tools\KITTI\original_kitti\data_object_image_2\training\image_' num2str(cam)]);
% calib_dir = 'C:\EPFL_tools\KITTI\original_kitti\data_object_calib\training\calib';

% options
root_dir = 'G:\Mon Drive\Colab Notebooks\DLAV GDrive\KITTI\data';

% get sub-directories
cam = 2; % 2 = left color camerag\
image_dir = fullfile(root_dir,['image_' num2str(cam)]);
label_dir = 'C:\EPFL_tools\KITTI\predicted\universe\data';
calib_dir = fullfile(root_dir,'calib');


% get the images labeled for this testing
nimages = length(dir(fullfile(label_dir, '*.txt')));
file_list = dir(fullfile(label_dir, '*.txt')); % get only the .txt files
file_list = file_list(~[file_list.isdir]); % exclude directories
file_names = {file_list.name}'; % extract the file names
fprintf("Number of images : %d\n", length(file_names))
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
    % prevent file "0" to be supressed
    if strcmp(file_names{i}, '000000.txt')
        current_name = '0';
    end
    % Replace the old name with the new name
    file_names{i} = current_name;
end

% set up figure
h = visualization('init',image_dir);

% main loop
img_idx=1;
while 1
    fprintf("\nFile index %d, i=%d \n", str2double(file_names{img_idx}), img_idx)
    % load projection matrix
    P = readCalibration(calib_dir,str2double(file_names{img_idx}),cam);

    % load labels
    objects = readLabels(label_dir,str2double(file_names{img_idx}));

    % visualization update for next frame
    visualization('update',image_dir,h,str2double(file_names{img_idx}),nimages,data_set, img_idx);
    % for all annotated objects do
    for obj_idx=1:numel(objects)

        % plot 2D bounding box
        drawBox2D(h,objects(obj_idx));

        % plot 3D bounding box
        [corners,face_idx] = computeBox3D(objects(obj_idx),P);
        orientation = computeOrientation3D(objects(obj_idx),P);
        drawBox3D(h, objects(obj_idx),corners,face_idx,orientation);

    end

    % force drawing and tiny user interface
    valid_key = 0;

    while ~valid_key
        valid_key = 1;
        waitforbuttonpress;
        key = get(gcf, 'CurrentKey');
        disp(key)
        
        switch lower(key)
            case 'q'
                break;
            case 'leftarrow'
                img_idx = max(img_idx - 1, 1);              % previous frame
            case 'rightarrow'
                img_idx = min(img_idx + 1, nimages - 1);    % next frame
            case 'space'
                img_idx = min(img_idx + 1, nimages - 1);    % next frame
            case 'uparrow'
                img_idx = min(img_idx + 100, nimages - 1);  % +100 frames
            case 'downarrow'
                img_idx = max(img_idx - 100, 1);            % -100 frames
            case 'n'
                img_idx = max(img_idx - 1000, 1);           % +1000 frames
            case 'm'
                img_idx = min(img_idx + 1000, nimages - 1); % -1000 frames
            otherwise
                valid_key = 0;
                continue;  % next frame
        end
        disp(valid_key)
    end


end

% clean up
close all;
