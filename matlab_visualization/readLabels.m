function objects = readLabels(label_dir,img_idx)
% parse input file
fid = fopen(sprintf('%s/%06d.txt',label_dir,img_idx),'r');

try
    C   = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
    labels_with_score = true;
catch
    try
        C   = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
        labels_with_score = false;
    catch
        disp("error with label file, incorrect number of entries")
        exit(1);
    end
end
fclose(fid);

% for all objects do
objects = [];

for o = 1:numel(C{1})

    % extract label, truncation, occlusion
    lbl = C{1}(o);                   % for converting: cell -> string
    objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...
    objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
    objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
    objects(o).alpha      = C{4}(o); % object observation angle ([-pi..pi])

    % extract 2D bounding box in 0-based coordinates
    objects(o).x1 = C{5}(o); % left
    objects(o).y1 = C{6}(o); % top
    objects(o).x2 = C{7}(o); % right
    objects(o).y2 = C{8}(o); % bottom

    % extract 3D bounding box information
    objects(o).h    = C{9} (o); % box width
    objects(o).w    = C{10}(o); % box height
    objects(o).l    = C{11}(o); % box length
    objects(o).t(1) = C{12}(o); % location (x)
    objects(o).t(2) = C{13}(o); % location (y)
    objects(o).t(3) = C{14}(o); % location (z)
    objects(o).ry   = C{15}(o); % yaw angle

    if labels_with_score
        objects(o).score = C{16}(o);% confidence score
    end

    %     if objects(o).truncation == -1
    %        objects(o).level = 0;
    %     end
    height = objects(o).y2 - objects(o).y1;
    if height >= 40 && objects(o).truncation <= 0.15 && objects(o).occlusion <= 0
        objects(o).level = 1;
    elseif height >= 25 && objects(o).truncation <= 0.3 && objects(o).occlusion <= 1
        objects(o).level = 2;
    elseif height >= 25 && objects(o).truncation <= 0.5 && objects(o).occlusion <= 2
        objects(o).level = 3;
    else
        objects(o).level = 4;
    end
end
