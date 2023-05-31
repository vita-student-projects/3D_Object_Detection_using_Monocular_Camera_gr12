function drawBox2D(h,object)

% set styles for object type
type_col = {'g','y','r','w', 'b','m','k'};
bigFatBoxes = false;

% draw regular objects
if ~strcmp(object.type,'DontCare')

  % show rectangular bounding boxes
  pos = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
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
  
  if bigFatBoxes == true
      rectangle('Position',pos,'EdgeColor',type_col{type},...
                'LineWidth',3,'LineStyle','parent',h(1).axes)
      rectangle('Position',pos,'EdgeColor','b', 'parent', h(1).axes)
  else
      rectangle('Position',pos,'EdgeColor',type_col{type}, 'parent', h(1).axes, 'LineWidth', 1)
  end

  % draw label
  label_text = sprintf('%s\n%1.0f %%',object.type,object.score*100);
  x = (object.x1+object.x2)/2;
  y = object.y1;
  text(x,max(y-5,40),label_text,'color',type_col{type},...
       'BackgroundColor','k','HorizontalAlignment','center',...
       'VerticalAlignment','bottom','FontWeight','bold',...
       'FontSize',8,'parent',h(1).axes);
     
% draw don't care regions
else
  
  % draw dotted rectangle
  pos = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
  rectangle('Position',pos,'EdgeColor','c',...
            'LineWidth',2,'LineStyle','-','parent',h(1).axes)
end
