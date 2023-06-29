"""
Phantoms.TrabeculaeVoronoi

A voronoi-seeding-based trabecular bone phantom.

Revision April 2021.

Authors:  Qian Cao, Xin Xu, Nada Kamona, Qin Li

"""

import numpy as np
# Delunary to test for colinearlity
from scipy.spatial import Voronoi, Delaunay
# import optimesh # May be a good idea for CVT post perturbation
from itertools import chain # list unpacking

from skimage.draw import line_nd
import scipy
from scipy.ndimage import binary_dilation

# distributions for seeding and perturbations
from scipy.stats import multivariate_normal

# import raster_geometry as rg

# import utils

#%% Utilities

def Polar2CartesianEllipsoid(Phi, Lambda, r, h, N):
    """
    Sample (x,y,z) from r=(Phi, Lambda, h)
    
    https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversions

    Note that this is NOT a uniform sampling of the ellipsoid.

    Parameters
    ----------
    Phi : np.ndarray
        Angle from X in XY.
    Lambda : np.ndarray
        Angle from XY plane.
    h : np.ndarray
    N

    Returns
    -------
    Pxyz : points in XYZ

    Notes
    -----
    This is NOT a uniform sampling of the ellipsoid.

    """
    raise(NotImplemented,"Currently supports sampleSphere")
    
def Polar2CartesianSphere(r, Theta, Phi):
    """
    Sample from sphere solid.
    
    https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Parameters
    ----------
    r : np.ndarray (1,) floats
        Distance from origin.
    Theta : np.ndarray (1,) floats
        Azimuthal angle in radians.
    Phi : np.ndarray (1,) floats
        Polar angle in radians.

    Returns
    -------
    np.ndarray[N,3]

    Notes
    -----
    This is NOT a uniform sampling of the ellipsoid.

    """
    
    assert r.ndim == 1
    assert Theta.ndim == 1
    assert Phi.ndim == 1
    
    x = r*np.sin(Theta)*np.cos(Phi)
    y = r*np.sin(Theta)*np.sin(Phi)
    z = r*np.cos(Theta)
    
    return np.array([x,y,z]).T

def MinMaxNorm(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def setEdgesZero(volume):
    """
    Sets edges of the volume to zero (for isosurface to generate a closed mesh.)

    Parameters
    ----------
    volume

    Returns
    -------
    volume

    """
    
    volume[0,:,:] = 0; volume[-1,:,:] = 0;
    volume[:,0,:] = 0; volume[:,-1,:] = 0;
    volume[:,:,0] = 0; volume[:,:,-1] = 0;
    
    return volume

def sampleUniformZeroOne(size, randState=None):
    """
    Samples a uniform distribution in [0,1), for generating distributions numerically.

    Parameters
    ----------
    size : tuple of ints
        Dimensions of uniform distribution.
    randState : int or None
        Seed used by the random generator. None is used.

    Returns
    -------
    randomArray

    """
    
    # Generate a random array for perturbation from [0,1)
    if randState is not None:
        r = np.random.RandomState(randState)
        rarr = r.random(size)
    else:
        rarr = np.random.random(size)
    
    return rarr

#%% Trabecular Phantom

def makeSeedPointsCartesian(Sxyz, Nxyz):
    """
    Generate perturbed seed points for voronoi tessellation

    Parameters
    ----------
    Sxyz : tuple or list of floats
        Size of VOI along each dimension (e.g. mm).
    Nxyz : tuple or list of integers
        Number of points along each dimension.

    Returns
    -------
    points: np.ndarray[N,3]
    
    The VOI is zero-centered. 
    
    Sxyz includes a half-increment padding along each dimension.

    """
    
    # Convert to nd.array
    Sxyz, Nxyz = np.array(Sxyz), np.array(Nxyz)
    
    assert len(Sxyz) == 3, "Dimension Error: Size of VOI must be 3"
    assert len(Nxyz) == 3, "Dimension Error: Number of seed points in the VOI must be 3"
    
    # Increment along each dimension
    Ixyz = Sxyz / Nxyz
    
    # Create array coordinates
    start = -Sxyz/2 + Ixyz/2
    end = +Sxyz/2 - Ixyz/2
    Cxyz = [np.linspace(start[dim],end[dim],Nxyz[dim]) for dim in range(3)]
    
    # Create meshgrid of points
    xyz = np.meshgrid(*Cxyz)
    points = np.array([xyz[dim].flatten() for dim in range(3)]).T
    
    return points

def perturbSeedPointsCartesianUniformXYZ(points, Rxyz, dist="sphere", randState=None):
    """
    Perturb points in XY, and then perturb Z. And then performs CVT and returns 
    Voronoi points.

    Parameters
    ----------
    points : np.ndarray[N,3]
        Cartesian seed points.
    Rxyz : scalar or (tuple or list) of floats
        Radius of maximum shifts in X, Y and Z.
    Sxyz : tuple or list of floats
        Size of VOI along each dimension (e.g. mm). Needed to filter out vertices out of extent.

    Returns
    -------
    voi : Voronoi object
    
    """
    
    # Number of points
    Np = points.shape[0]
    
    # Generate a random array for perturbation from [0,1)
    rarr = sampleUniformZeroOne((Np,3), randState=randState)
    
    # # Generate a random array for perturbation from [0,1)
    # if randState is not None:
    #     r = np.random.RandomState(randState)
    #     rarr = r.random((Np,3))
    # else:
    #     rarr = np.random.random((Np,3))
        
    # Sample perturbation from sphere
    R = rarr[:,0]*Rxyz
    Theta = rarr[:,1]*2*np.pi
    Phi = MinMaxNorm(rarr[:,2], -np.pi/2, np.pi/2)
    Pxyz = Polar2CartesianSphere(R,Theta,Phi)
    
    # Perturb points in XY
    ppoints = points + Pxyz
    
    return ppoints

def perturbSeedPointsEllipsoidUniformXYZ(points, Rxyz, randState=None):
    """
    https://stackoverflow.com/questions/24513304/uniform-sampling-from-a-ellipsoidal-confidence-region
    
    """
    
    pass
    
    # # Perturb points in XY
    # ppoints = points    # Perturb points in XY
    #     ppoints = points + Pxyz
    
    # return ppoints


def perturbSeedPointsGaussianXYZ(points, sigmaXYZ, randState=None):
    
    # Number of points
    Np = points.shape[0]
    
    # Generate Gaussian perturbation
    covariance = np.diag(sigmaXYZ**2)
    Pxyz = np.random.multivariate_normal(mean=(0,0,0), cov=covariance, random_state = randState)
    
    # Perturb points in XY
    ppoints = points + Pxyz
    
    return ppoints
    
def applyVoronoi(ppoints, Sxyz):
    """
    Apply Voronoi Tessellation to (perturbed) points, and compute index of 
    vertices within the VOI.

    Parameters
    ----------
    ppoints : np.ndarray [N,3]
        List of perturbed points.
    Sxyz : tuple
        Size of VOI along each dimension (e.g. mm). Needed to filter out vertices out of extent.

    Returns
    -------
    vor : scipy.spatial.qhull.Voronoi
        Voronoi tessellation object.
    ind : np.ndarray integers
        Index of vertices within VOI.

    """
    
    # Voronoi Points in XY, retry on QHull error
    while True:
        try:
            vor = Voronoi(ppoints)
        except:
            raise("QHull Error, retrying")
            continue
        break
    
    # TODO Centroidal Voronoi Tessellation
    # https://github.com/nschloe/optimesh (Du et al)
    
    # Extract index of Voronoi vertices within the VOI extent
    Sxyz = np.array(Sxyz)
    ind = np.nonzero(np.prod(np.abs(vor.vertices) < Sxyz/2, axis=1))[0]
    
    return vor, ind

def findUniqueEdgesAndFaces(vor, ind, maxEdgePerFace=8):
    """
    Compute unique edges for each Voronoi cell

    Parameters
    ----------
    vor : scipy.spatial.qhull.Voronoi
        Voronoi tessellation object.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
    ind : np.ndarray integers
        Index of vertices within VOI.
    maxEdgePerFace : integer
        Maximum edges per face.

    Returns
    -------
    uniqueEdges, uniqueFaces

    """
    
    # iterate through list and picks out pairs of adjacent elements
    def roll2(data_list):
        return [[data_list[x],data_list[x+1]] for x in range(len(data_list)-1)]
        
    # Filter data_list and keep only unique elements (which are also lists)
    def uniq(data_list):
        # https://stackoverflow.com/questions/3724551/python-uniqueness-for-list-of-lists
       return [list(x) for x in set(frozenset(x) for x in data_list)]
        
    # Vertices which have been excluded (outside the extent of VOI)
    ind_exclude = set(np.arange(len(vor.vertices)))-set(np.array(ind))
    ind_exclude.add(-1) # -1 means Vornoi vertex outside the Voronoi Diagram
    
    # Exclude ridges ("faces", list of list of Voronoi vertices) containing excluded vertices
    ridge_ind = [x for x in range(len(vor.ridge_vertices))
                if not bool(set(vor.ridge_vertices[x]) & set(ind_exclude))]
    ridge = [vor.ridge_vertices[x] for x in ridge_ind]
    # ridge_numverts = [len(vor.ridge_vertices[x]) for x in ridge_ind] # DEBUG only

    # TODO: Condense vertices with coplanar vertices (most faces)
    # See https://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d
    # varargout = mergeCoplanarFaces(nodes, varargin)
    
    # Extract edges from faces
    ridge_circ = [vor.ridge_vertices[x] + [vor.ridge_vertices[x][0]]
                  for x in ridge_ind] # This is a copy of ridge
    edges = [roll2(x) for x in ridge_circ]
    # https://stackoverflow.com/questions/45816549/flatten-a-nested-list-using-list-unpacking-in-a-list-comprehension
    edges = list(chain.from_iterable(edges))
    
    # Apply uniqueness to faces and edges
    uniqueFaces = uniq(ridge)
    uniqueEdges = uniq(edges)
    
    return uniqueEdges, uniqueFaces

def getEdgeVertices(vertices, edges):
    """
    Get vertices selected by ind, after converting to np.ndarray.

    Parameters
    ----------
    vertices : np.ndarray (Nverts,3)
        Coordinates of vertices.
    edges : [[indStart, indEnd],...,[indStart, indEnd]] (Nedges,2) integers
        Vertex indices of the starting and end points.

    Returns
    -------
    vertices : np.ndarray (Nedges,2,3)
        Edge vertices.

    """
    edges, vertices = np.array(edges), np.array(vertices)
    edgeVertices = vertices[edges,:]
    
    return edgeVertices

def computeEdgeCosine(edgeVertices, direction = (0,0,1)):
    """
    Compute direction cosine with of edges with points in vertices, along a
    specified direction.

    Parameters
    ----------
    edgeVertices : np.ndarray (Nverts,2,3)
        Coordinates of edge vertices.
    direction: tuple
        Direction the cosine is computed against.

    Returns
    -------
    cosines : np.ndarray (Nedges)
        cosine with respect to direction.

    Notes
    -----
    Use np.abs(cosines) to convert (-) to (+) direction.

    """
    direction = np.array(direction)
    edgeVectors = edgeVertices[:,1,:] - edgeVertices[:,0,:] # (Nedges,3)

    cosines = np.dot(edgeVectors,direction) \
        / np.linalg.norm(edgeVectors,axis=1) / np.linalg.norm(direction)
    
    return cosines

def filterEdgesRandomUniform(uniqueEdges, retainFraction = 0.8, randState=None):
    """
    Drop random edges uniformly throughout the entire VOI
    
    Parameters
    ----------
    uniqueEdges
    retainFraction : default=0.8
    randState : default=None

    Returns
    -------
    uniqueEdgesRetain

    """
    # TODO does other edge-drop schemes belong here? Maybe use another function for that
    
    uniqueEdges = np.array(uniqueEdges)
    
    Nedges = uniqueEdges.shape[0]
    
    # if isinstance(retainFraction, float):
    Nretain = np.round(Nedges*retainFraction).astype(int)
    
    if randState is not None:
        r = np.random.RandomState(seed=randState)
        retainInd = r.choice(Nedges, Nretain, replace=False)
    else:
        retainInd = np.random.choice(Nedges, Nretain, replace=False)
        
    uniqueEdgesRetain = [uniqueEdges[x] for x in retainInd]
        
    return uniqueEdgesRetain, retainInd

def getFaceVertices(vertices, faces):
    """
    Get vertices selected by ind, after converting to np.ndarray.

    Parameters
    ----------
    vertices : np.ndarray (Nverts,3)
        Coordinates of vertices.
    faces : [[v0, v1,...],...,[v0, v1, ...]] list of lists
        Vertex indices of each face. Where each face may have different numbers of vertices

    Returns
    -------
    faceVertices : list of np.ndarray (FaceVerts,3)
        Face vertex coordinates.

    """
    
    faceVertices = [vertices[x,:] for x in faces]
    
    return faceVertices

def computeFaceNormals(faceVertices):
    """
    Compute face normals for a list of face vertex coordinates.

    Parameters
    ----------
    faceVertices : list of np.ndarrays (FaceVerts,3)'s
        Face vertex coordinates. Note: Each face may have different number of coplanar vertices.

    Returns
    -------
    faceNormals : np.ndarray (Nfaces, 3)

    """
    
    def computeNormal(verts):
        # verts : nd.nparray (Npoints,3)
        # TODO since coplanar, just use the first 3 vertices maybe revise this to
        # select vertives evenly distributed around a polygon
        verts = verts[:3,:]
        
        vec1 = verts[1,:] - verts[0,:]
        vec2 = verts[2,:] - verts[0,:]
        
        vecn = np.cross(vec2,vec1)
        
        normal = vecn / np.linalg.norm(vecn)
        
        return normal
    
    faceNormals = [computeNormal(x) for x in faceVertices]
    
    return faceNormals

def computeFaceCentroids(faceVertices):
    """
    Compute face centroid for a list of face vertex coordinates.

    Parameters
    ----------
    faceVertices : list of np.ndarrays (FaceVerts,3)'s
        Face vertex coordinates. Note: Each face may have different number of coplanar vertices.

    Returns
    -------
    faceCentroids : np.ndarray (Nfaces, 3)

    """
    def computeCentroid(verts):
        return np.mean(verts,axis=0)
    
    faceCentroids = [computeCentroid(x) for x in faceVertices]
    
    return faceCentroids

def computeFaceAreas(faceVertices):
    """
    Compute face area for a list of face vertex coordinates.
    
    https://math.stackexchange.com/questions/3207981/caculate-area-of-polygon-in-3d

    Parameters
    ----------
    faceVertices : list of np.ndarrays (FaceVerts,3)'s
        Face vertex coordinates. Note: Each face may have different number of coplanar vertices.

    Returns
    -------
    faceAreas : np.ndarray (Nfaces, 3)

    """
    def computeArea(verts):
        # compute area for a list of coplanar 3D vertices
        
        v0 = verts[0,:]
        vk = verts[2:,:] - v0
        vj = verts[1:-1,:] - v0
        
        # ||sum(vk x vj)||/2
        area = np.linalg.norm(np.sum(np.cross(vk,vj,axis=1),axis=0))/2
        
        return area
    
    faceAreas = [computeArea(x) for x in faceVertices]
    
    return faceAreas

def filterFacesRandomUniform(uniqueFaces, retainFraction, randState=None):
    
    # TODO does other edge-drop schemes belong here?
    
    Nfaces = len(uniqueFaces)
    
    # if isinstance(retainFraction, float):
    Nretain = np.round(Nfaces*retainFraction).astype(int)
    
    if randState is not None:
        r = np.random.RandomState(randState)
        retainInd = r.choice(Nfaces, Nretain, replace=False)
    else:
        retainInd = np.random.choice(Nfaces, Nretain, replace=False)
        
    uniqueFacesRetain = [uniqueFaces[x] for x in retainInd]
        
    return uniqueFacesRetain, retainInd

def convertAbs2Array(vertices, voxelSize, volumeSizeVoxels):
    # convert absolute coordinates (e.g. in mm) to array coordinates (array indices)
    # In absolute coordinates, origin is in the center, in array coordinates, origin is at "top left" corner.
    
    if type(voxelSize) is tuple:
        voxelSize = np.array(voxelSize)
        
    if type(volumeSizeVoxels) is tuple:
        volumeSizeVoxels = np.array(volumeSizeVoxels)
        
    shiftOriginToCorner = voxelSize * volumeSizeVoxels / 2
    
    # Note, array coordinates start at 0.5
    verticesArray = (vertices + shiftOriginToCorner) / voxelSize - 0.5
    
    return verticesArray

def convertArray2Abs(vertices, voxelSize, volumeSizeVoxels):
    # convert array coordinates to absolute coordinates
    # In absolute coordinates, origin is in the center, in array coordinates, origin is at "top left" corner.
    
    if type(voxelSize) is tuple:
        voxelSize = np.array(voxelSize)
        
    if type(volumeSizeVoxels) is tuple:
        volumeSizeVoxels = np.array(volumeSizeVoxels)
        
    shiftOriginToCorner = voxelSize * volumeSizeVoxels / 2
    
    # Note, array coordinates start at 0.5
    verticesAbs = (vertices + 0.5) * voxelSize - shiftOriginToCorner
    
    return verticesAbs

def drawLine(volume, vertices, edgeVertexInd, value):
    # Vertices should be in array coordinates (corner origin, unit = voxels)
    
    start = vertices[edgeVertexInd[0],:]
    end = vertices[edgeVertexInd[1],:]
    
    lin = line_nd(start, end, endpoint = True)

    volume[lin] = value
    
def flood_fill_hull(image):
    # This is depricated TODO: remove
    # This is taken from:
    #   https://stackoverflow.com/questions/46310603/how-to-compute-convex-hull-image-volume-in-3d-numpy-arrays/46314485#46314485
    # A modified version may be used (TODO: test if this one is faster):
    #   https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    
    return out_img, hull

def full_triangle(a, b, c):
    # Fill trangle using bresenham_line (changed to using line_nd instead)
    # https://stackoverflow.com/questions/60901888/draw-triangle-in-3d-numpy-array-in-python
    # a, b and c are TUPLES of coordinates of the triangle
    # returns SET of tuples
    
    def line_nd_set(a, b):
        ab = line_nd(a, b, endpoint=True) # tuple of coordinates
        return {tuple(x) for x in np.array(ab).T} # convert to tuple of points
    
    ab = line_nd_set(a, b) # set of vertices in ab
    abc = ab.copy()
    for x in ab:
        abc = abc.union(line_nd_set(c, x))
    
    return abc
        
def full_surface(verts):
    # Array of vertices corresponding to a coplanar face
    # returns SET of integer coordinates
    
    # Define points of the triangle that needs to be filled
    v0 = verts[0,:] # a
    vk = verts[2:,:] # b
    vj = verts[1:-1,:] # 
    
    surf = set()
    for jj in range(vj.shape[0]): # for the number of triangles
        tri = full_triangle(tuple(v0), 
                            tuple(vk[jj,:]), 
                            tuple(vj[jj,:]))
        surf = surf.union(tri)
    
    return surf

def drawFaces(volume, verticesArray, faceVertexInd, values=None): # TODO: Test this augmented version
    # Draws multiple faces
    # Vertices should be in array coordinates (corner origin, unit = voxels)
    
    if values is None: # create binary volume
        for faceInd in range(len(faceVertexInd)):
            print(f"{faceInd}/{len(faceVertexInd)}")
            faceVerts = verticesArray[faceVertexInd[faceInd],:].astype(int)
            surf = full_surface(faceVerts)
            surf = tuple(np.array(list(surf)).T) # convert from set of points to tuple of XYZ coordinates
            volume[surf] = 1
            
        volume = (volume>0).astype(np.uint8) # TODO: this might not be necessary
        
    else: # create volume with volumes assigned
        for faceInd in range(len(faceVertexInd)):
            print(f"{faceInd}/{len(faceVertexInd)}")
            faceVerts = verticesArray[faceVertexInd[faceInd],:].astype(int)
            surf = full_surface(faceVerts)
            surf = tuple(np.array(list(surf)).T) # convert from set of points to tuple of XYZ coordinates
            volume[surf] = np.maximum(volume[surf],values[faceInd])
            
    return volume

def makeSkeletonVolume(vertices, edgeInds, faceInds, voxelSize, volumeSizeVoxels):
    # Converts vertex list, edge vertex index and face vertex index to volume.
    
    volume = np.zeros(volumeSizeVoxels, dtype=np.uint8)
    
    # Convert vertices to array-coordinates
    verticesArray = convertAbs2Array(vertices, voxelSize, volumeSizeVoxels)
    
    # TODO: finish wrapper class for makeSkeletonVolumeEdgesAndFaces
    
    return volume

def makeSkeletonVolumeEdges(vertices, edgeInds, voxelSize, volumeSizeVoxels, values=None):
    # Converts vertex list, edge vertex index and face vertex index to volume.
    # This function only assigns voxels from edges.
    
    if values is None: # Create a binary volume
        volume = np.zeros(volumeSizeVoxels, dtype=np.uint8)
    else: # Create a volume with the same type as values
        volume = np.zeros(volumeSizeVoxels, dtype=values.dtype)
    
    # Convert vertices to array-coordinates
    verticesArray = convertAbs2Array(vertices, voxelSize, volumeSizeVoxels)
    
    edgeInds = np.array(edgeInds)
    
    if values is None:
        for ii in range(edgeInds.shape[0]):
            drawLine(volume, verticesArray, edgeInds[ii,:], 1)
    else:
        for ii in range(edgeInds.shape[0]):
            drawLine(volume, verticesArray, edgeInds[ii,:], values[ii])
    
    return volume

def makeSkeletonVolumeFaces(vertices, faceInds, voxelSize, volumeSizeVoxels, values=None):
    # invoke drawFace
    # vertices: all vertices from vor.vertices
    # adding values argument to drawFace, values must be none or have same len as faceInds
    
    if values is None: # Create a binary volume
        volume = np.zeros(volumeSizeVoxels, dtype=np.uint8)
    else: # Create a volume with the same type as values
        volume = np.zeros(volumeSizeVoxels, dtype=values.dtype)
        
    print(volume.shape)
    
    # Convert vertices to array-coordinates
    verticesArray = convertAbs2Array(vertices, voxelSize, volumeSizeVoxels)
    
    # invoke drawFace
    drawFaces(volume, verticesArray, faceInds, values)
    
    return volume

def dilateVolumeSphereUniform(volume, radius):
    """
    Binary dilation of skeletal mask with spherical mask with specified radius.
    
    """
    
    diam = np.arange(-radius,radius+1)
    xx, yy, zz = np.meshgrid(diam, diam, diam)
    sphere = (xx**2+yy**2+zz**2<=radius**2)
    
    volumeDilated = binary_dilation(volume,sphere)
    
    return volumeDilated

def cdfRosconi(cdfThickness=np.linspace(0,1,1000), 
               alpha=1.71e11, beta=8.17, gamma=55.54):
    """
    TODO: Not Yet Implemented
    
    * Input to this function has units of mm for default parameters.
    ** Default values of alpha, beta and gamma derived from:
       Rosconi et al. Quantitative approach to the stochastics of bone remodeling. 2012.
     
    thickness - range of thicknesses (in mm) considered in the pdf calculation.
    """
    
    # Thickness distribution (using default parameters, thickness is in mm)
    def distributionRosconi(t, alpha=alpha, beta=beta, gamma=gamma):
        return alpha*(t**beta)*np.exp(-gamma*t)
    
    pdf = distributionRosconi(cdfThickness, alpha, beta, gamma)
    pdf = pdf / np.sum(pdf)
    cdf = np.cumsum(pdf)
    
    return cdfThickness, cdf

def sampleRosconi(cdfThickness, cdf, size=None, randState=None):
    """
    TODO: Not Yet Implemented
    
    Sample from a Rosconi distribution.
    Random state of uniform variable generator defined by randState
    
    Parameters
    ----------
    t
        thickness values corresponding to cdf.
    cdf : 1D array
        1D array of the corresponding Rosconi cdf.
    size
        shape of the output array (independent cdf).
    
    """
    
    # Generate a random array for perturbation from [0,1)
    rarr = sampleUniformZeroOne(size, randState=randState)
    
    # Interpolate cdf (linear)
    return np.interp(rarr, cdf, cdfThickness)

def thicknessOrientationDisributionFunctionKinney(theta, phi):
    """
    TODO: Not Yet Implemented
    
    Returns mean trabecular thickness per orientation.
    
    Kinney et al. An orientation distribution function for trabecular bone.
    https://reader.elsevier.com/reader/sd/pii/S8756328204003734?token=C6F2ADE8594DCAB20514450AC22BE062BC81E235B0D2AE2AFEC24A86078B22526169261B2108B6CCF49685FB96F36B00&originRegion=us-east-1&originCreation=20210924183910
    
    Figure 5.

    Returns
    -------
    None.

    """

    return None
    
def massOrientationDistributionFunctionKinney(theta, phi):
    """
    TODO: Not Yet Implemented
    
    Compute relative mass of trabeculae given orientation.
    
    Kinney et al. An orientation distribution function for trabecular bone.

    Returns
    -------
    None.

    """
    
    return None

# def dilateVolumeSphereRosconi(volume, voxelSize, alpha=1.71e11, beta=8.17, gamma=55.54):
#     # Dilate trabecular structure based on poisson distribution
#     # radius in units of voxels

#     # create sphere
#     radius = (T/voxelSize).astype(int) # [mm]/[mm/voxel]
#     diam = np.arange(-radius,radius+1)
#     xx, yy, zz = np.meshgrid(diam, diam, diam)

#     return volumeDilated

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    plt.ion()
    
    print('Running example for TrabeculaeVoronoi')
    
    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (10,10,10), (10,10,10) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 1.
    edgesRetainFraction = 0.5
    facesRetainFraction = 0.1
    dilationRadius = 3 # (voxels)
    randState = 123 # for repeatability
    
    # Parameters for generating phantom volume
    volumeSizeVoxels = (200,200,200)
    voxelSize = np.array(Sxyz) / np.array(volumeSizeVoxels)
    
    # Generate faces and edges
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=randState)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    uniqueEdges, uniqueFaces = findUniqueEdgesAndFaces(vor, ind)
    
    # Compute edge cosines
    edgeVertices = getEdgeVertices(vor.vertices, uniqueEdges)
    edgeCosines = computeEdgeCosine(edgeVertices, direction = (0,0,1))
    
    # Compute face properties
    faceVertices = getFaceVertices(vor.vertices, uniqueFaces)
    faceAreas = computeFaceAreas(faceVertices)
    faceCentroids = computeFaceCentroids(faceVertices)
    faceNormals = computeFaceNormals(faceVertices)
    
    # Filter random edges and faces
    uniqueEdgesRetain, edgesRetainInd = filterEdgesRandomUniform(uniqueEdges, 
                                                                 edgesRetainFraction, 
                                                                 randState=randState)
    uniqueFacesRetain, facesRetainInd = filterFacesRandomUniform(uniqueFaces, 
                                                                 facesRetainFraction, 
                                                                 randState=randState)
    
    volumeEdges = makeSkeletonVolumeEdges(vor.vertices, uniqueEdgesRetain, voxelSize, volumeSizeVoxels)
    volumeFaces = makeSkeletonVolumeFaces(vor.vertices, uniqueFacesRetain, voxelSize, volumeSizeVoxels)
    
    # Uniform dilation
    volumeDilated = dilateVolumeSphereUniform(np.logical_or(volumeEdges,volumeFaces), dilationRadius)
    # volumeDilated = dilateVolumeSphereUniform(volumeEdges, dilationRadius)
    
    # Testing code for drawFaces
    # faceVertexInd = uniqueFacesRetain
    # vertices = vor.vertices
    # volume = np.zeros(volumeSizeVoxels, dtype=np.uint16)
    # verticesArray = convertAbs2Array(vertices, voxelSize, volumeSizeVoxels)
    
    # Visualize all edges
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for ii in range(edgeVertices.shape[0]):
    #     ax.plot(edgeVertices[ii,:,0],edgeVertices[ii,:,1],edgeVertices[ii,:,2],'b-')
    
    # # Visualize a face
    # face = uniqueFaces[-5]
    # fv = vor.vertices[face,:]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(fv[:,0],fv[:,1],fv[:,2],'bo')
