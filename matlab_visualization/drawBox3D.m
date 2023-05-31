function drawBox3D(h,object,corners,face_idx,orientation)

  
  
  % set styles for object type
type_col = {'g','y','r','w', 'b','m','k'};
bigFatBoxes = false;

% draw regular objects
if ~strcmp(object.type,'DontCare')

  % show rectangular bounding boxes
%   pos = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
  type = 1;
  if strcmp(object.type, 'Car')
      type = 2;
  elseif strcmp(object.type, 'Pedestrian') || strcmp(object.type, 'Person_sitting')
      type = 3;
  elseif strcmp(object.type, 'Cyclist')
      type = 4;
  elseif strcmp(object.type, 'Van') || strcmp(object.type, 'Truck')
      type = 5;
  elseif strcmp(object.type, 'Tram')
      type = 6;
  end

  % draw projected 3D bounding boxes
  if ~isempty(corners)
    for f=1:4
        if bigFatBoxes == true
            line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
            [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
            'parent',h(2).axes, 'color',type_col{type},...
            'LineWidth',2,'LineStyle');
            line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
               [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
               'parent',h(2).axes,'color','b','LineWidth',0.5);
        else
            line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
               [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
               'parent',h(2).axes,'color', type_col{type},'LineWidth',1);
        end
    end
  end
  
  % draw orientation vector
  if ~isempty(orientation)
    line([orientation(1,:),orientation(1,:)]+1,...
         [orientation(2,:),orientation(2,:)]+1,...
         'parent',h(2).axes,'color','w','LineWidth',4);
    line([orientation(1,:),orientation(1,:)]+1,...
         [orientation(2,:),orientation(2,:)]+1,...
         'parent',h(2).axes,'color','k','LineWidth',2);
  end
end
