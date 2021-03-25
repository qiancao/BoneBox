function [objImg_wplates, numPlatesFilled, idxPlatesFilled, conn_plates, idxLongPlatesFilled] = RandAddPlates_qli_test_201908_2(V, allfaces, boneProps, plateArea_threshold, prob)
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
% Last modifed by Qin Li 4/30/2019
% additional modification: (particular setup for vertebrae bone)
% - to vectorize it
% - to add another angle for determining plate directionality
% - to introduce a directionality field (e.g., pre-specified, can be obtained from real bone data - with interpolation)

%% Check input variables
minArgs = 5;
maxArgs = 5;
debug_on = 0;
if debug_on == 1
    disp('in debug mode; to change, modify the parameter in the code')
end
%narginchk(minArgs, maxArgs) % Gives an error if more or less arguments are given

uniq_faces = allfaces.uniq_faces;

%% Determine Bone Type

if isempty(plateArea_threshold)
    plateArea_threshold = 5*boneProps.deltaX ^2;
end

if isempty(prob)
    prob = 0.5;
end

%get_angle = @(u,v) atan2(norm(cross([u(:)],[v(:)])), dot(u,v)); % find angle between two vectors (0-pi)
get_angle = @(u,v) atan2(sqrt(sum(cross(u,v).^2,2))', dot(u',v')); % find angle between two vectors (0-pi); u v can be matrix
% each row of u and v are two vectors to compare


% face_angle: angle between face-ortho and vv; abs(face-angle)-pi/2: angle between face and vv
% face_vn: angle between face-orth and vvn;
% todo: plateAngle_threshold could be different
plateAngle_threshold = deg2rad(20); % keep vertical
plateAngle_threshold_2 = deg2rad(20); % keep all
plateAngle_threshold_3 = deg2rad(20);
%fillPlate = @(area, face_angle) ((area<plateArea_threshold))|(abs(abs(face_angle)-pi/2)< plateAngle_threshold);
fillPlate = @(area, face_angle, face_angle_orth) (abs(abs(face_angle)-pi/2)< plateAngle_threshold & abs(abs(face_angle_orth)-pi/2)< plateAngle_threshold_2);

plateAngle_threshold_a = deg2rad(40);
fillPlate_a = @(area, face_angle, face_angle_orth) (abs(abs(face_angle)-pi/2)< plateAngle_threshold_a ...
    & abs(abs(face_angle)-pi/2)> plateAngle_threshold & abs(face_angle_orth)< plateAngle_threshold_2);
vv1 = [0 0 1]'; % angle to be align
vv2 = [1 1 0]'; % angle the vn of the face to be align
vv3 = [1 -1 0]';


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

plates_weights = [];

[p1, p2] = meshgrid([-50:49],[-50:49]);
plateWeightsFunc = sqrt(p1.^2+p2.^2);

idxPlatesFilled = [];

area_all = allfaces.area;
vn_all = allfaces.face_orth_vector; %face-orth vector: The cross product between
%two 3-D vectors produces a new vector that is perpendicular to both.

centroid_all = allfaces.centroid;



% ** the angle between the z direction and the plane can determine
% the directionality of the plane
vv1 = vv1(:)'; % make it row-vector
vv1_all = repmat(vv1, size(vn_all,1), 1);
face_angle_1_all = get_angle(vv1_all, vn_all); % angle between the face-orth vector to vv (the difference between this and pi/2 determines whether the face aligns with direction vv (0: perfect align))
vv2 = vv2(:)';
vv2_all = repmat(vv2, size(vn_all,1), 1);
face_angle_2_all = get_angle(vv2_all, vn_all); % angle between the face-orth vector to vvn
vv3 = vv3(:)';
vv3_all = repmat(vv3, size(vn_all,1), 1);
face_angle_3_all = get_angle(vv3_all, vn_all);

original_index = 1:length(uniq_faces);
% filtered by direction
alpha_1 = abs(abs(face_angle_1_all)-pi/2);
alpha_2 = abs(abs(face_angle_2_all)-pi/2);
alpha_3 = abs(abs(face_angle_3_all)-pi/2);

main_face_to_keep = (alpha_1<deg2rad(30) & (alpha_2<deg2rad(30) | alpha_3<deg2rad(30)));



normalize_row = @(x) rdivide(x, sqrt(sum(x.^2,2)));
vn_all = normalize_row(vn_all);

d_thresh = 0.2; c_thresh = 0.8; z_thresh = 5;
face_not_visited = ones(size(original_index));
new_index = original_index;
conn_plates = [];

for i = 1:length(uniq_faces)
    if main_face_to_keep(i) == 1 & face_not_visited(i) == 1
        nb_not_visited = sum(face_not_visited.*main_face_to_keep);
        tmp_index = find(face_not_visited.*main_face_to_keep);
        cos_between_faces = abs(dot(repmat(vn_all(i,:), nb_not_visited, 1)', vn_all(tmp_index,:)')');
        ctr_diff = repmat(centroid_all(i,:), nb_not_visited, 1)-centroid_all(tmp_index,:);
        dist_between_xy = sqrt(sum(ctr_diff(:,1:2).^2,2));
        dist_between_z = abs(ctr_diff(:,3));
        connected_plates_index = tmp_index(find(dist_between_xy<d_thresh & cos_between_faces>c_thresh ...
            & dist_between_z<z_thresh));
        new_index(connected_plates_index) = i;
        face_not_visited(connected_plates_index) = 0;
        if length(connected_plates_index)>1
            conn_plates = [conn_plates; [i, length(connected_plates_index)]];   
        end
    end
end

isFill_longplates = zeros(size(alpha_1));
for k = 1:length(conn_plates)
    isFill_longplates(new_index == conn_plates(k,1)) = 1;
end


%prob_1 = exp(-(alpha_1.^2)/2/deg2rad(10)^2);
%prob_2 = exp(-(alpha_2.^2)/2/deg2rad(20)^2);
%prob_3 = exp(-(alpha_3.^2)/2/deg2rad(20)^2);

isFill_random = rand(size(isFill_longplates))<1/6*sum(isFill_longplates)/length(uniq_faces);
isFill = isFill_random | isFill_longplates;




idxPlatesFilled = find(isFill==1);     numPlatesFilled = length(idxPlatesFilled);
idxLongPlatesFilled = find(isFill_longplates==1);

for i = 1:numPlatesFilled
    
    tmp = uniq_faces(idxPlatesFilled(i),:);
    poly_idx = tmp(tmp>0);
    
    
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

plates_pointsIDX = round(plates_pointsIDX);
objImg_wplates = zeros(numel(boneProps.MeshX),1);
if ~isempty(plates_pointsIDX)
    idxpoints_1d = ((plates_pointsIDX(:,3))-1)*size(boneProps.MeshX,1)*size(boneProps.MeshX,2)+...
        ((plates_pointsIDX(:,2))-1)*size(boneProps.MeshX,1) + (plates_pointsIDX(:,1));
    
    %objImg_wplates(idxpoints_1d)=plates_weights; % set all indeces from above to 1
    idxpoints_1d_bounded = idxpoints_1d(find(idxpoints_1d<numel(boneProps.MeshX) & idxpoints_1d>0));
    objImg_wplates(idxpoints_1d_bounded)=1;
end
objImg_wplates = reshape(objImg_wplates, size(boneProps.MeshX,1), size(boneProps.MeshX,2), size(boneProps.MeshX,3));
