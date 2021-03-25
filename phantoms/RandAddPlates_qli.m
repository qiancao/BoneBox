function [objImg_wplates, numPlatesFilled, idxPlatesFilled] = RandAddPlates_qli(V, uniq_faces, boneProps, plateArea_threshold, prob)
%% Summary:
% Function to randomly add plates to bone model, depending on bone type, angle, and plate area threshold
%
% Inputs: - V = voronoi seeds list from voronoiDiagram, xyz coordinates
%         - uniq_edges = a list of mx2, where m is the total number of edges. Each row has the indices of two end points of the edge
%         - uniq_faces = a matrix each row (no-zero) corresponds to a uniq face (limited by number of edges)
%         - boneProps: a struct containing the following fields:
%               boneProps.bonetype = string specifying bone type, ex: 'isotropic, femur, vertebral
%               boneProps.MeshX = mesh grid for the object with resolution deltaX & deltaZ
%               boneProps.deltaX = resolution in X and Y directions (mm/pixel)
%               boneProps.deltaZ = resolution in Z directions (mm/pixel)
%               boneProps.dimX = bone object dimension in the X and Y directions (pixels)
%               boneProps.dimZ = bone object dimension in the Z directions (pixels)
%         - plateArea_threshold = plate area threshold, plates added to the object will have area small or equal to it.
%
% Outputs:- objImg_wplates = object 3D matrix with plates added. The edges and plates are weighted (NOT BINARY)
%         - numPlatesFilled = final number of plates filled
%         - idxPlatesFilled = plates index in uniq_faces that are filled (for plotting purpose)
%
% Last modifed by Qin Li 4/24/2019
%% Check input variables
minArgs = 5;
maxArgs = 5;
debug_on = 0;
if debug_on == 1
    disp('in debug mode; to change, modify the parameter in the code')
end
%narginchk(minArgs, maxArgs) % Gives an error if more or less arguments are given

%% Determine Bone Type

if isempty(plateArea_threshold)
    plateArea_threshold = 5*boneProps.deltaX ^2;
end

if isempty(prob)
    prob = 0.5;
end

get_angle = @(u,v) atan2(norm(cross([u(:)],[v(:)])), dot(u,v)); % find angle between two vectors (0-pi)

switch boneProps.bonetype
    case 'isotropic'
        plateAngle_threshold = deg2rad(45); % need to have plates more vertical and horizontal
%         fillPlate = @(area, face_angle) ((area<plateArea_threshold))|(abs(face_angle-pi/2)<plateAngle_threshold) ...
%             | (abs(face_angle-pi/2)>(pi/2-plateAngle_threshold));
        %fillPlate = @(area, face_angle) ((area<plateArea_threshold))|(abs(face_angle-pi/2)<plateAngle_threshold); % original
        fillPlate = @(area, face_angle) ((area<plateArea_threshold));
%         fillPlate = @(area, face_angle) (abs(face_angle-pi/2)<plateAngle_threshold) ...
%             | (abs(face_angle-pi/2)>(pi/2-plateAngle_threshold));
        vv = [0 0 1]';
   
    case 'vertebral'
        plateAngle_threshold = deg2rad(20); % keep vertical 
        %fillPlate = @(area, face_angle) ((area<plateArea_threshold))|(abs(abs(face_angle)-pi/2)< plateAngle_threshold);
        fillPlate = @(area, face_angle) (abs(abs(face_angle)-pi/2)< plateAngle_threshold);
        vv = [0 0 1]';
    case 'femur'
        plateAngle_threshold = deg2rad(15); %not sure yet****
        fillPlate = @(area, face_angle) (abs(face_angle-pi/2)<plateAngle_threshold);
        vv = [0 0 1]';
    case 'spherical_control'
        fillPlate = @(area, face_angle) ((area<plateArea_threshold));
        vv = [0 0 1]';
end

% debugging plots
if debug_on
    for i = 1:size(uniq_faces,1) % go row by row (each row is a face)
        % Create a convex hull that contains all the vertices in the current
        % cell (a region)
        tmp = uniq_faces(i,:);
        poly_idx = tmp(tmp>0);
        poly_idx = [poly_idx, poly_idx(1)];
        plot3(V(poly_idx,1), V(poly_idx,2), V(poly_idx,3),'g')
        hold on
    end
end
%% randomly add plates
V(1,:) = [-10 -10 -10];
plates_pointsIDX = []; 
numPlatesFilled = 0;
idx_bounded_C = []; 

% plateWeightsFunc(1:5000) = exp(-1*(linspace(1,10,5000)))+1; 
% plateWeightsFunc(5001:10000)=plateWeightsFunc(5000:-1:1);
% plateWeightsFunc = reshape(plateWeightsFunc,100,100);
plates_weights = [];

[p1, p2] = meshgrid([-50:49],[-50:49]);
plateWeightsFunc = sqrt(p1.^2+p2.^2);

idxPlatesFilled = [];

for i = 1:size(uniq_faces,1) % go row by row (each row is a face)
    % Create a convex hull that contains all the vertices in the current
    % cell (a region)
    tmp = uniq_faces(i,:);
    poly_idx = tmp(tmp>0);                                               
    vx = V(poly_idx(1),:);
    vy = V(poly_idx(2),:);
    vz = V(poly_idx(3),:);
    vyx = vy-vx; vzx = vz-vx; % vectors between the three coordinates
    vn = cross(vzx, vyx); %face-orth vector: The cross product between 
                          %two 3-D vectors produces a new vector that is perpendicular to both.

    % ** the angle between the z direction and the plane can determine
    % the directionality of the plane

    %area = 1/2*norm(vn); % returns the Euclidean norm of vector v
    area = trapz(V(poly_idx,1),V(poly_idx,2)); % use trapzoid integral to get the area (double-check)
    face_angle = get_angle(vv, vn); % angle between the face-orth vector to vertical direction
    
    %isFill = ((area<plateArea_threshold))|(abs(face_angle-pi/2)<plateAngle_threshold);
    isFill = fillPlate(area, face_angle);
    
    if rand(1)<prob && isFill % random probablity check
        % if prob = isFill = 0, this loop will never be excuted
        idxPlatesFilled = [idxPlatesFilled i];
        numPlatesFilled = numPlatesFilled + 1;
        
        %facepoints = fill_facet(vx,vy,vz,4); % xyz coordinates for each vertix/point in the plane
        facepoints = fill_poly(V, poly_idx);
        
        % tmp is the pixel number associated with those coordinates
        tmp = round(facepoints./repmat([boneProps.deltaX boneProps.deltaX boneProps.deltaZ],size(facepoints,1),1)+0.5*boneProps.dimX);
        %tmp = tmp(:,[2 1 3]); % switch 1st and 2nd columns
        tmp(:,3) = tmp(:,3)-0.5*boneProps.dimX;
        
        % get the points that are within the defined obj dimensions
        idxtmp = find(tmp(:,1)>0 & tmp(:,2)>0 & tmp(:,3)>0 & ...
            tmp(:,1)<=boneProps.dimX & tmp(:,2)<=boneProps.dimX & tmp(:,3)<=boneProps.dimZ);
        plates_pointsIDX = [plates_pointsIDX; tmp(idxtmp,:)]; % store all points that make up the plates.
        if ~isempty(idxtmp)
            plates_weights = [plates_weights; plateWeightsFunc(round(([1:length(idxtmp)]-1)*9999/max([length(idxtmp)-1,1])+1))'];
        end
        if debug_on
            hold on
            fill3(V(poly_idx,1), V(poly_idx,2), V(poly_idx,3),'r','facealpha',0.3)
                       
        end
    end
end


% objImg_wplates = zeros(numel(boneProps.MeshX),1);
% if ~isempty(plates_pointsIDX)
%     idxpoints_1d = (plates_pointsIDX(:,3)-1)*size(boneProps.MeshX,1)*size(boneProps.MeshX,2)+...
%         (plates_pointsIDX(:,2)-1)*size(boneProps.MeshX,1) + plates_pointsIDX(:,1);
%        
%     %objImg_wplates(idxpoints_1d)=plates_weights; % set all indeces from above to 1
%     objImg_wplates(int64(idxpoints_1d))=1;
% end
% objImg_wplates = reshape(objImg_wplates, size(boneProps.MeshX,1), size(boneProps.MeshX,2), size(boneProps.MeshX,3));  

plates_pointsIDX = round(plates_pointsIDX);
objImg_wplates = zeros(numel(boneProps.MeshX),1);
if ~isempty(plates_pointsIDX)
    idxpoints_1d = ((plates_pointsIDX(:,3))-1)*size(boneProps.MeshX,1)*size(boneProps.MeshX,2)+...
        ((plates_pointsIDX(:,2))-1)*size(boneProps.MeshX,1) + (plates_pointsIDX(:,1));
       
    %objImg_wplates(idxpoints_1d)=plates_weights; % set all indeces from above to 1
    objImg_wplates(idxpoints_1d)=1;
end
objImg_wplates = reshape(objImg_wplates, size(boneProps.MeshX,1), size(boneProps.MeshX,2), size(boneProps.MeshX,3));  



end