function [ V, C, idx_bounded, pointsCoordinates ] = makeVertebralBone_test_201908_1(cvtfunc, sliceHeight, pointDistr_func, radius0, randR)
%% Summary of this function
%   Make irregular bone sample based Wang et al: Generate regular seed point and then randomize them with control by proablity sphere
%   Use Voronoi tesselation to generate cells from the poitns generated.
%   Find bounded cells only. 
%       
%   Inputs: - cvtfunc.generator_num = number of seeds points
%           - cvtfunc.iteration_num = number of iteratiosn for cvt_square_nonuniform.m, between 20 and 50
%           - cvtfunc.samplepoints_num = for cvt_square_nonuniform.m, %A value of 1,000 is too low.  
%                              A value of 1,000,000 is somewhat high
%           - sliceHeight = bone object height
%           - radius0 = bone object radius, or half width. Make radius half the height
%           - randR: radius of the sphere that controls the randomness
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
%   Last modified by Qin Li, 4/22/2019: trying different seed point generating methodology

%% Generate points coordinates
if nargin<5
    randR = 0.3;
end
xx = []; yy=[]; zz=[];

tmpV = ((2*radius0)^2*sliceHeight);
delta = (tmpV/cvtfunc.generator_num)^(1/3);

zoom_factor = 2.5; KK_factor = 1.1;
z_xy_density_ratio = 0.7;

%KK = poissrnd(cvtfunc.generator_num/M); % generates random numbers from the Poisson distribution with mean parameter lambda
M =  round(sliceHeight/delta*z_xy_density_ratio); %number of layers in z direction
KK = round(zoom_factor^KK_factor*cvtfunc.generator_num/M*1.1/z_xy_density_ratio);    %number of seed points on each plan * 1.1
voronoiPoints = cvt_square_nonuniform(KK, cvtfunc.samplepoints_num, pointDistr_func, cvtfunc.iteration_num);

%densify_func = @(x) -1/2*x^2;
xx_old = voronoiPoints(:,1)*radius0*zoom_factor; yy_old = voronoiPoints(:,2)*radius0*zoom_factor;
tbd = 2;
M = tbd*M;
delta_z = sliceHeight/M;
for k = 1:M
    if k<M/2+1
        dfdx = @(x) -x; dfdy = @(y) -y; delta_t = 0.03;
    else
        dfdx = @(x) +x; dfdy = @(y) +y; delta_t = 0.03;
    end
    xx_new = xx_old + delta_t*dfdx(xx_old);
    yy_new = yy_old + delta_t*dfdy(yy_old)-0.1;
    
    xx = [xx; xx_new];
    yy = [yy; yy_new];
    xx_old = xx_new; yy_old = yy_new;
    
    zz_new = zeros(size(xx_new));
    xpartition = [min(xx_new)-1, -radius0*2/3, -radius0/3, 0, radius0/3, radius0*2/3, max(xx_new)+1];
    ypartition = [min(yy_new)-1, -radius0*2/3, -radius0/3, 0, radius0/3, radius0*2/3, max(yy_new)+1];
    for i = 1:6
        for j = 1:6
            xlower = xpartition(i); xhigher = xpartition(i+1);
            ylower = ypartition(j); yhigher = ypartition(j+1);
            idx = find(xx_new>=xlower & xx_new<xhigher & yy_new>=ylower & yy_new < yhigher);
            zz_new(idx) = ((k-1)+rand(1)+0.3*rand(length(idx),1))*delta_z;
        end
    end
    zz = [zz; zz_new];
end



idx_seedpoint_bounded = find(abs(xx)<=radius0 & abs(yy)<=radius0);
l = 0;
while(l<3)
    nbidx_ratio = length(idx_seedpoint_bounded)/cvtfunc.generator_num/1.1/tbd
    
    if nbidx_ratio < 0.95
        shrink_alpha = sqrt(nbidx_ratio);
        xx = xx*shrink_alpha;
        yy = yy*shrink_alpha;
    elseif nbidx_ratio > 1.05
        expand_alpha = sqrt(nbidx_ratio);
        xx = xx*expand_alpha;
        yy = yy*expand_alpha;
    end
    idx_seedpoint_bounded = find(abs(xx)<=radius0 & abs(yy)<=radius0);
    l = l+1;
end

xx = xx(idx_seedpoint_bounded); yy = yy(idx_seedpoint_bounded); zz = zz(idx_seedpoint_bounded);

allTheta = rand(size(xx))*pi; allPhi = rand(size(xx))*2*pi; allR = randR*delta*rand(size(xx));
xx = xx+allR.*sin(allTheta).*sin(allPhi);
yy = yy+allR.*sin(allTheta).*cos(allPhi);
%zz = zz+allR.*cos(allTheta);

N = length(xx);

pointsCoordinates = [xx(:), yy(:), zz(:)];
pointsCoordinates = pointsCoordinates(rand(N,1)<0.9,:);

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
idx_unboundedV = find(abs(V(:,1))>radius0 | abs(V(:,2))>radius0 ...
    | V(:,3)> sliceHeight | V(:,3)<0);

% find bounded verticies
idx_bounded = setdiff(1:length(V),idx_unboundedV);

