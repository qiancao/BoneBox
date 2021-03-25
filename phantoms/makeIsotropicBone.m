function [ V, C, idx_bounded, pointsCoordinates] = makeIsotropicBone( ...
    cvtfunc, sliceHeight, radius0, Flag_pointsDistr, pointDistr_func)
%% Summary of this function
%   Make Isotropic bone sample: homogenous/uniform in all directions.
%   Generate seed points coordinates using one of the methods in Flag_poitnsDistr 
%   Use Voronoi tesselation to generate cells from the poitns generated.
%   Find bounded cells only. 
%       
%   Inputs: - cvtfunc.generator_num = number of seeds points
%           - cvtfunc.iteration_num = number of iteratiosn for cvt_square_nonuniform.m, between 20 and 50
%           - cvtfunc.samplepoints_num = for cvt_square_nonuniform.m, %A value of 1,000 is too low.  
%                              A value of 1,000,000 is somewhat high
%           - sliceHeight = bone object height
%           - radius0 = bone object radius, or half width. Make radius half the height
%           - Flag_pointsDistr = 
%               0 = distribute points across M slices (XY planes), then generate Z randomly within each slice (FSU code)
%               0.5 = XYZ coordinates are generated randomly.
%               1 = distribute ALL points in X-Y plane (cvt_square_nonuniform), generate Z coordinates randomly (FSU code)    
%               2 = distribute points in a regular lattice, then perturb (NOT WORKING) ***
%           - pointDistr_func = custom pointDistr_func, otherwise use default.
%
%   Outputs:- V = voronoi points xyz coordinates
%           - C = voronoi cells
%           - idx_bounded = indeces of bounded points in V
%           - pointsCoordinates = xyz coordinates of the points used to create voronoi points
%
%   Last modified by Nada Kamona, 08/09/2018
%% Check Inout arguments
if nargin == 5
    if isempty(pointDistr_func) % if null
        pointDistr_func = @(x,y)10;
    end
elseif nargin <4
    pointDistr_func = @(x,y)10; %Isotropic from Qin
    Flag_pointsDistr = 0;
elseif nargin < 5
    pointDistr_func = @(x,y)10; %Isotropic from Qin
end
%% Generate points coordinates
xx = []; yy=[]; zz=[];

if Flag_pointsDistr == 1 
    voronoiPoints = cvt_square_nonuniform(cvtfunc.generator_num, cvtfunc.samplepoints_num, pointDistr_func, cvtfunc.iteration_num);
    xx = [xx; voronoiPoints(:,1)]*radius0; yy = [yy; voronoiPoints(:,2)]*radius0;
    %xx1 = rand(c0,1)*2*R0-R0; xx2 = rand(c0,1)*2*R0-R0;
    zz = rand(cvtfunc.generator_num,1)*sliceHeight;

elseif Flag_pointsDistr == 2 % DOESN'T WORK
    
%     x = linspace(0,cvtfunc.generator_num*2*radius0-radius0,(cvtfunc.generator_num)^(1/3));
    x = linspace(0,1*2*radius0,(cvtfunc.generator_num)^(1/3));
    %z = linspace(0,generator_num*2*radius0-radius0,(generator_num)^(1/3));
    z = linspace(0,1*sliceHeight,(cvtfunc.generator_num)^(1/3));
    % define regulatr mesh
    [X,Y,Z] = meshgrid(x,x,z);
    
    randFactorX = 0.02;
    randFactorY = 0.02;
    randFactorZ = 0.035;

    X_shifted = arrayfun(@(x) x+randn(1)*randFactorX,X);
    Y_shifted = arrayfun(@(x) x+randn(1)*randFactorY,Y);
    Z_shifted = arrayfun(@(x) x+randn(1)*randFactorZ,Z);
    
    xx = abs(X_shifted(:)); yy = abs(Y_shifted(:)); zz=abs(Z_shifted(:));

elseif Flag_pointsDistr == 0.5
    %xx = [xx; voronoiPoints(:,1)]*radius0; yy = [yy; voronoiPoints(:,2)]*radius0;
    xx = rand(cvtfunc.generator_num,1)*2*radius0-radius0; 
    yy = rand(cvtfunc.generator_num,1)*2*radius0-radius0;
    zz = rand(cvtfunc.generator_num,1)*sliceHeight;
   
else
    M = 5;
    stepsize = sliceHeight/M;
    for k = 1:M
        KK = poissrnd(cvtfunc.generator_num/M); % generates random numbers from the Poisson distribution with mean parameter lambda
        voronoiPoints = cvt_square_nonuniform(KK, cvtfunc.samplepoints_num, pointDistr_func, cvtfunc.iteration_num);
        xx = [xx; voronoiPoints(:,1)*radius0]; 
        yy = [yy; voronoiPoints(:,2)*radius0];
        %xx1 = rand(c0,1)*2*R0-R0; xx2 = rand(c0,1)*2*R0-R0;
        zz = [zz; rand(KK,1)*stepsize + (k-1)*stepsize];
    end
end

pointsCoordinates = [xx, yy, zz];

%% Voronoi Tessellation
% Add a matlab version checkpoint
if ~strcmp(version, '7.5.0.338 (R2007b)' )
    % V voronoi vertices; C cells
    dt = delaunayTriangulation(pointsCoordinates); % Original, but not supported
    [V,C] = voronoiDiagram(dt); % Original, but not supported
else
    [V,C] = voronoin(pointsCoordinates);
end

%% Bounded Points
% find unbounded verticies
idx_unboundedV = find(sqrt(V(:,1).^2+V(:,2).^2)>radius0 ...
    | V(:,3)> sliceHeight | V(:,3)<0);

% find bounded verticies
idx_bounded = setdiff(1:length(V),idx_unboundedV);

end
