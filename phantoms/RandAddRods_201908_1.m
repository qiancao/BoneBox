function [uniq_edges_final, isPrune, abs_cos] = RandAddRods_201908_1( V, uniq_edges, radius0, alpha_angleKeep, ...
    beta_angleDrop, femurParameters, bonetype)
%% Summary of this function
% Function to randomly drop edges from bone model, depending on bone type and other set parameters:
%
% Notes: - For isotropic case, poisson distributiion is used to control the number of edges connected for each seed,
%          with lamda = 4, which could be changed if needed in the code below.
%
% Inputs: - V= voronoi seeds list from voronoiDiagram, xyz coordinates
%         - uniq_edges = a list of mx2, where m is the total number of edges. Each row has the indices of two end points of the edge
%         - radius0 = radius of the bone sample, or basically half the dimension of width/height
%         - Variables for dropping edges by angle, needed for get_angle function handle
%           - beta_angleDrop=  increase beta -> less edges kept
%           - alpha_angleKeep= angles to keep 100%, increase alpha ->  more edges kept
%         - fumerParameters:
%               femurParameters.w0 = 1;
%               femurParameters.w0sigma = 0.25;
%               femurParameters.w0beta = 10; %smaller beta -> less points
%               femurParameters.fmu = 1;
%               femurParameters.fsigma = 0.31;% original is simga = 0.25, which controls how wide the hollow part is
%         - bonetype = string specifying bone type, ex: 'isotropic, femur, vertebral
%
% Outputs:- uniq_edges_final = final list of edges after pruning
%
% Last modified by Qin Li 09/28/2018
%% Check input arguments
% minArgs = 3;
% maxArgs = 7;
% 
% narginchk(minArgs, maxArgs) % Gives an error if more or less arguments are given
% 
% if nargin<4
%     alpha_angleKeep = pi/10;
%     beta_angleDrop = 20;
%     bonetype = 'isotropic';
% elseif nargin<5
%     beta_angleDrop = 20;
%     bonetype = 'isotropic';
% elseif nargin<6
%     bonetype = 'isotropic';
% end

%% Determine bone type

% define a function to find the key angles/vector at a single point
% assign a function to the probability of certain angle to be dropped out
get_angle = @(u,v) atan2(norm(cross([u(:)],[v(:)])), dot(u,v)); % find angle between two vectors (0-pi)
%theta = deg2rad(10);

switch bonetype
    case 'isotropic'      
        %**** Drop by Radius Variables ****
        %w0 = 1; sigma1 = 0.1; beta = 1; %smaller beta -> less points
        %wfun = @(rr) w0/sigma1/sqrt(2*pi)*integral(f, -Inf,rr);        
        %weightfun = @(rr) w0/sigma1/sqrt(2*pi); % Bone weights function, Hui Li 2013
        
        lambda = 4; % average number of edges per vertix
        %**** Drop by Angle Variables ****
        % cal%culate:
        %   1) angle between vv and the z-axis, 
        %   2) angle between vv and the xy-plane
        get_2vectsAngle = @(vv) min([get_angle([0 0 1]', vv), get_angle([vv(1),vv(2),0]', vv)]); % for the isotropic case
        %get_2vectsAngle = @(vv) get_angle([0 0 1]', vv); % for the isotropic case
    case 'spherical_control' 
        lambda = 4
        get_2vectsAngle = @(vv) min([get_angle([0 0 1]', vv), get_angle([vv(1),vv(2),0]', vv)]); % for the isotropic case

    case 'vertebral'
        % ADD CODE HERE
        
        
        %**** Drop by Radius Variablies ****
        
        
        %**** Drop by Angle Variablies ****
        % cal%culate:
        %   1) angle between vv and the z-axis to keep more of the vertical edges and less of the horizontal edges, 
        get_2vectsAngle = @(vv) get_angle([0 0 1]', vv); % measure the angle with the z-axis
        
    case 'femur'
        %**** Drop by Radius Variablies ****
        f = @(u) exp(-(u-femurParameters.fmu).^2/2/(femurParameters.fsigma^2)); % cumulative normal distribution
        weightfun = @(rr) femurParameters.w0/femurParameters.w0sigma/sqrt(2*pi)*integral(f, -Inf,rr); % Bone weights function, Hui Li 2013
        
        %**** Drop by Angle Variablies ****
        % cal%culate:
        %   1) angle between vv and the z-axis, 
        %   2) angle between vv and the xy-plane
        get_2vectsAngle = @(vv) min([get_angle([0 0 1]', vv), get_angle([vv(1),vv(2),0]', vv)]); % for the isotropic case
end

%% randomly drop edges (by radius)

%Initialize variables
numEdges = length(uniq_edges); % number of unique edges
magNormal_min = zeros(numEdges,1); % magnitude of the normal vector (min)
weightedEdges_temp = zeros(numEdges,1); 
magNormal_max = zeros(numEdges,1); % magnitude of the normal vector (max)

if strcmp(bonetype,'isotropic') || strcmp(bonetype,'spherical_control')
    % METHOD 1: 
%     numPoints = length(V); %number of points
%     vertPoissRand = poissrnd(lambda,numPoints,1); %poisson random number for each vertix
%     probDeletionFunc = [];
%     for thisPoint = 1:numPoints
%         p = V(thisPoint,:); %take the coordinate of current point
%         tmp_idx = find(uniq_edges(:,1)==thisPoint); %edges connected to current point
%         if ~isempty(tmp_idx)
%             probDeletionFunc = [probDeletionFunc; ((10-(-0))*rand(length(tmp_idx),1)-0)< vertPoissRand(thisPoint)];
%         end
%     end
%     for thisEdge = 1:numEdges % for every edge (two points each)
%         %find the coordinates (voronoi) of the two points of the current edge
%         p1 = V(uniq_edges(thisEdge,1),1:2); p2 = V(uniq_edges(thisEdge,2),1:2);
%         aa = norm(p1); bb = norm(p2); % returns the Euclidean norm of vector v (magnitude)
%         magNormal_min(thisEdge) = min(aa, bb); magNormal_max(thisEdge) = max(aa,bb);
%     end
%     uniq_edge_1 = uniq_edges(find(probDeletionFunc==1 & magNormal_max<1.1*radius0),:);
    
%     % METHOD 2:
%     numPoints = length(V); %number of points
%     vertPoissRand = poissrnd(lambda,numPoints,1); %poisson random number for each vertix
%     edges2Keep = [];
%     for thisPoint = 1:numPoints
%         p = V(thisPoint,:); %take the coordinate of current point
%         tmp_idx = find(uniq_edges(:,1)==thisPoint); %edges connected to current point (QL: not exact, as some vertices appears in uniq__edges(:,2))
%         if ~isempty(tmp_idx)
%             % pick edges equal to vertPoissRand(thisPoint)
%             if length(tmp_idx)>= round(vertPoissRand(thisPoint))
%                 %tmp_idx_selected = randi([1,length(tmp_idx)],round(vertPoissRand(thisPoint)),1);
%                 %edges2Keep = [edges2Keep; tmp_idx(tmp_idx_selected)];
%                 tmp_idx_selected = randi([1,length(tmp_idx)],round(vertPoissRand(thisPoint)),1);
%                 tmp_idx_selected = ((10-(-0))*rand(length(tmp_idx_selected),1)-0)< vertPoissRand(thisPoint);
%                 edges2Keep = [edges2Keep; tmp_idx(find(tmp_idx_selected==1))];
%             else
%                 tmpProb = ((10-(-0))*rand(length(tmp_idx),1)-0)< vertPoissRand(thisPoint);
%                 edges2Keep = [edges2Keep; tmp_idx(find(tmpProb==1))];
%                 %edges2Keep = [edges2Keep; tmp_idx];
%             end
%         end
%     end
%     uniq_edge_1 = uniq_edges(edges2Keep,:);
    
    % Method 3: (QLi, to targeting 4:3:1 for 3-N, 4-N, 5-N; previous version does not account that every edge deletes, two vertices affected; and there are many vertices already smaller than targeted poission)
    numPoints = length(V);
    p_4n = 0.7; p_5n = 0.2; p_6n = 0.1;
    edges2Keep = []; edges2Exclude = [];
    uniq_edge_now = uniq_edges;
    for thisPoint =1:numPoints
        p = V(thisPoint,:); % take the coordinate of current point
        tmp_idx = find(uniq_edges(:,1) == thisPoint); 
        tmp_nb = sum(uniq_edge_now(:) == thisPoint); % number of edges connected to current point
        if tmp_nb>0
            if tmp_nb<=4
%                 if rand<1./3
%                     nb2Exclude = 1;
%                     tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
%                     idx_exclude = tmpComb(randi(size(tmpComb,1),1),:); 
%                     edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)]; 
%                 end
            elseif tmp_nb==5 |tmp_nb==6
                    nb2Exclude = 2;
                    tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
                   % idx_include = tmpComb(randi(size(tmpComb,1),1),:); 
                   % idx_exclude = setdiff(1:length(tmp_idx), idx_include);
                    idx_exclude = tmpComb(ceil(rand(1)*size(tmpComb,1)));%tmpComb(randi(size(tmpComb,1),1),:); 
                    %tmp_idx_selected = tmp_idx(idx_include);
                    %edges2Keep = [edges2Keep; tmp_idx_selected(:)];
                    edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];
%             elseif tmp_nb==6
%                     tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),2)); % get rid of two edge
%                     idx_include = tmpComb(randi(size(tmpComb,1),1),:); idx_exclude = setdiff(1:length(tmp_idx), idx_include);
%                     tmp_idx_selected = tmp_idx(idx_include);
%                     edges2Keep = [edges2Keep; tmp_idx_selected(:)];
%                     edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];
            elseif tmp_nb >= 7
                    nb2Exclude = ceil(rand(1)*2);%randi(2,1);
                    tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
                    idx_exclude = tmpComb(ceil(rand(1)*size(tmpComb,1)));%tmpComb(randi(size(tmpComb,1),1),:); 
                    edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];
                    %tmpComb = nchoosek(1:length(tmp_idx),max(0,length(tmp_idx)-(tmp_nb-tmpnumber))); % get rid of one edge
                    %idx_include = tmpComb(randi(size(tmpComb,1),1),:); idx_exclude = setdiff(1:length(tmp_idx), idx_include);
                    %tmp_idx_selected = tmp_idx(idx_include);
                    %edges2Keep = [edges2Keep; tmp_idx_selected(:)];
            end
        end
        uniq_edge_now = uniq_edges(setdiff(1:length(uniq_edges),edges2Exclude),:);
    end
    uniq_edge_1 = uniq_edge_now;
    for i = 1:max(uniq_edges(:))
        ttt_1(i) = sum(uniq_edge_now(:)==i);
        ttt(i) = sum(uniq_edges(:)==i);
    end
    sum(ttt_1<3)
    sum(ttt_1==3)
    sum(ttt_1==4)
    sum(ttt_1==5)
    sum(ttt_1>5)
    
elseif strcmp(bonetype,'femur')
    
    numPoints = length(V);
    p_4n = 0.7; p_5n = 0.2; p_6n = 0.1;
    edges2Keep = []; edges2Exclude = [];
    uniq_edge_now = uniq_edges;
    for thisPoint =1:numPoints
        p = V(thisPoint,:); % take the coordinate of current point
        tmp_idx = find(uniq_edges(:,1) == thisPoint); 
        tmp_nb = sum(uniq_edge_now(:) == thisPoint); % number of edges connected to current point
        if tmp_nb>0
            if tmp_nb<=4
            elseif tmp_nb==5 |tmp_nb==6
                    nb2Exclude = 2;
                    tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
                    idx_exclude = tmpComb(ceil(rand(1)*size(tmpComb,1)));%tmpComb(randi(size(tmpComb,1),1),:); 
                    edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];
            elseif tmp_nb >= 7
                    nb2Exclude = ceil(rand(1)*2);%randi(2,1);
                    tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
                    idx_exclude = tmpComb(ceil(rand(1)*size(tmpComb,1)));%tmpComb(randi(size(tmpComb,1),1),:); 
                    edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];
            end
        end
        uniq_edge_now = uniq_edges(setdiff(1:length(uniq_edges),edges2Exclude),:);
    end
    uniq_edges = uniq_edge_now;
 
    numEdges = length(uniq_edges);
    magNormal_min = zeros(numEdges,1); 
    weightedEdges_temp = zeros(numEdges,1); 
    magNormal_max = zeros(numEdges,1); % magnitude of the normal vector (max)
    
    for thisEdge = 1:numEdges % for every edge (two points each)
        %find the coordinates (voronoi) of the two points of the current edge
        p1 = V(uniq_edges(thisEdge,1),1:2); p2 = V(uniq_edges(thisEdge,2),1:2);
        % the normal of each edge is the radius - the distance of that edge to
        % the center of the circle.
        aa = norm(p1); bb = norm(p2); % returns the Euclidean norm of vector v (magnitude)
        magNormal_min(thisEdge) = min(aa, bb); magNormal_max(thisEdge) = max(aa,bb);
        weightedEdges_temp(thisEdge) = weightfun(magNormal_min(thisEdge)); %assign a weight to the min normal 
        % ^ this is NOT useful for ISOTROPIC
    end
    % calculate the maximum edge weight, given the max Radius is R0
    maxWeight = weightfun(radius0); %maybe modified
    weightedEdges = weightedEdges_temp.*(weightedEdges_temp<=maxWeight) + maxWeight.*(weightedEdges_temp>maxWeight);
    % edges with weights <=wmax will be zero in the first term, and then
    % corrected to have the wmax as a weight instead in the second term

    % probability deletion function of the edge weight (w_i)
    probDeletionFunc = rand(numEdges,1)> exp(-femurParameters.w0beta*weightedEdges./maxWeight); % smaller r->smaller ww ->bigger exp ->less probability to be 1
    uniq_edge_1 = uniq_edges(find(probDeletionFunc==1 & magNormal_max<1.1*radius0),:);
    disp('hi');
    
elseif strcmp(bonetype,'vertebral')
    numPoints = length(V);
    p_4n = 0.7; p_5n = 0.2; p_6n = 0.1;
    edges2Keep = []; edges2Exclude = [];
    uniq_edge_now = uniq_edges;
    
    all_edges_vector = V(uniq_edges(:,1),:)-V(uniq_edges(:,2),:); 
    vn_all = repmat([0 0 1]',1, length(uniq_edges));
    norm_1 = sqrt(sum(all_edges_vector'.^2, 1));
    norm_2 = sqrt(sum(vn_all.^2, 1));
    cos_edge_and_vertical = dot(all_edges_vector', vn_all)./norm_1./norm_2; %cosine of the angle between P1 and P2
    abs_cos = abs(cos_edge_and_vertical)';
    isPrune = zeros(length(uniq_edges),1); % to record whether an edge has been pruned
    isPrune = (exp(-(1-abs_cos).^2/2/deg2rad(30)^2) < rand(length(uniq_edges),1)); % cos close to 1 indicate vertical --> less likely to be pruned
    
    edges2Exclude = find(isPrune==1);
    uniq_edge_now = uniq_edges(setdiff(1:length(uniq_edges), edges2Exclude),:);
      
    
    for thisPoint =1:numPoints
        p = V(thisPoint,:); % take the coordinate of current point
        tmp_idx = find(uniq_edges(:,1) == thisPoint); 
        tmp_nb = sum(uniq_edge_now(:) == thisPoint); % number of edges connected to current point
        if tmp_nb>0
            if tmp_nb<=4
            elseif tmp_nb==5 |tmp_nb==6
                    nb2Exclude = 1;
                    tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
                    idx_exclude = tmpComb(ceil(rand(1)*size(tmpComb,1)));%tmpComb(randi(size(tmpComb,1),1),:); 
                    edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];

            elseif tmp_nb >= 7
                    nb2Exclude = randi(2,1);
                    tmpComb = nchoosek(1:length(tmp_idx),min(length(tmp_idx),nb2Exclude)); % get rid of one edge
                    idx_exclude = tmpComb(ceil(rand(1)*size(tmpComb,1)));%tmpComb(randi(size(tmpComb,1),1),:); 
                    edges2Exclude = [edges2Exclude; tmp_idx(idx_exclude)];
            end
        end
        uniq_edge_now = uniq_edges(setdiff(1:length(uniq_edges),unique(edges2Exclude)),:);
    end
    uniq_edge_1 = uniq_edge_now;
    for i = 1:max(uniq_edges(:))
        ttt_1(i) = sum(uniq_edge_now(:)==i);
        ttt(i) = sum(uniq_edges(:)==i);
    end
    sum(ttt_1<3)
    sum(ttt_1==3)
    sum(ttt_1==4)
    sum(ttt_1==5)
    sum(ttt_1>5)  
    
    isPrune(unique(edges2Exclude)) = 1;
end

% %% randomly drop edges (by angles)
% 
% nbedge1 = length(uniq_edge_1);
% aa = zeros(nbedge1,1); 
% for thisEdge = 1:nbedge1
%     %find the coordinates (voronoi) of the two points of the current edge
%     p1 = V(uniq_edge_1(thisEdge,1),:); p2 = V(uniq_edge_1(thisEdge,2),:);
%     
%     % Find the vector between the two points
%     vv = p1-p2;
%         
%     aa(thisEdge) = get_2vectsAngle(vv); % NOTE: This can be controlled by the bone type
%     %aa(i) = min([get_a(aa1,vv) get_a(aa2,vv)]);% get_a(bb1,vv) get_a(bb2,vv)]);
% end
% aa = aa-alpha_angleKeep; 
% aa = aa.*(aa>0); % angles less than or equal to 0 will be set to zero
% 
% probDeletionFunc2 = rand(nbedge1,1)< exp(-beta_angleDrop*aa); % when aa = 0 -> keep the edge; increase aa -> less chance to keep the edge
% 
% uniq_edges_final = uniq_edge_1(find(probDeletionFunc2==1),:);

uniq_edges_final = uniq_edge_now;

end
