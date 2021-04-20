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

import utils

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
    if randState is not None:
        r = np.random.RandomState(randState)
        rarr = r.random((Np,3))
    else:
        rarr = np.random.random((Np,3))
        
    # Sample perturbation from sphere
    R = rarr[:,0]*Rxyz
    Theta = rarr[:,1]*2*np.pi
    Phi = utils.MinMaxNorm(rarr[:,2], -np.pi/2, np.pi/2)
    Pxyz = utils.Polar2CartesianSphere(R,Theta,Phi)
    
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
    
    Note: use np.abs(cosines) to convert (-) to (+) direction.

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
    uniqueEdges : TYPE
        DESCRIPTION.
    retainFraction : TYPE, optional
        DESCRIPTION. The default is 0.8.
    randState : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    uniqueEdgesRetain : TYPE
        DESCRIPTION.

    """
    # TODO does other edge-drop schemes belong here?
    
    Nedges = len(uniqueEdges)
    
    if isinstance(retainFraction, float):
        Nretain = np.round(Nedges*retainFraction).astype(int)
        
        if randState is not None:
            r = np.random.RandomState(randState)
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
    
    if isinstance(retainFraction, float):
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
    
    voxelSize, volumeSizeVoxels= np.array(voxelSize), np.array(volumeSizeVoxels)
    shiftOriginToCorner = voxelSize * volumeSizeVoxels / 2
    verticesArray = (vertices + shiftOriginToCorner) / voxelSize
    
    return verticesArray

def convertArray2Abs(vertices, voxelSize, volumeSizeVoxels):
    # convert array coordinates to absolute coordinates 
    # In absolute coordinates, origin is in the center, in array coordinates, origin is at "top left" corner.
    
    voxelSize, volumeSizeVoxels= np.array(voxelSize), np.array(volumeSizeVoxels)
    shiftOriginToCorner = voxelSize * volumeSizeVoxels / 2
    verticesAbs = vertices * voxelSize - shiftOriginToCorner
    
    return verticesAbs

def drawLine(volume, vertices, edgeVertexInd):
    # Vertices should be in array coordinates (corner origin, unit = voxels)
    
    start = vertices[edgeVertexInd[0],:]
    end = vertices[edgeVertexInd[1],:]
    
    lin = line_nd(start, end, endpoint = True)
    volume[lin] = 1

def drawFace(volume, vertices, faceVertexInd):
    # Vertices should be in array coordinates (corner origin, unit = voxels)
    
    pass

def makeSkeletonVolume(vertices, edgeInds, faceInds, 
                       voxelSize, volumeSizeVoxels):
    # Converts vertex list, edge vertex index and face vertex index to volume.
    
    volume = np.zeros(volumeSizeVoxels, dtype=np.uint16)
    
    # Convert vertices to array-coordinates
    verticesArray = convertAbs2Array(vertices, voxelSize, volumeSizeVoxels)
    
    # TODO finish this off here
    
    return volume

def makeSkeletonVolumeEdges(vertices, edgeInds, voxelSize, volumeSizeVoxels):
    # Converts vertex list, edge vertex index and face vertex index to volume.
    # This function only assigns voxels from edges.
    
    volume = np.zeros(volumeSizeVoxels, dtype=np.uint16)
    
    # Convert vertices to array-coordinates
    verticesArray = convertAbs2Array(vertices, voxelSize, volumeSizeVoxels)
    
    edgeInds = np.array(edgeInds)
    
    for ii in range(edgeInds.shape[0]):
        drawLine(volume, verticesArray, edgeInds[ii,:])
    
    return volume

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    plt.ion()
    
    print('Running example for TrabeculaeVoronoi')
    
    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (10,10,10), (5,5,5) # volume extent in XYZ, number of seeds along XYZ
    Rxyz = 0.5
    edgesRetainFraction = 0.5
    facesRetainFraction = 0.5
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
    faceNormas = computeFaceNormals(faceVertices)
    
    # Filter random edges and faces
    uniqueEdgesRetain, edgesRetainInd = filterEdgesRandomUniform(uniqueEdges, 
                                                                 edgesRetainFraction, 
                                                                 randState=randState)
    uniqueFacesRetain, facesRetainInd = filterFacesRandomUniform(uniqueFaces, 
                                                                 facesRetainFraction, 
                                                                 randState=randState)
    
    volume = makeSkeletonVolumeEdges(vor.vertices, uniqueEdges, voxelSize, volumeSizeVoxels)
    
    
    # # Visualize a face
    # face = uniqueFaces[-5]
    # fv = vor.vertices[face,:]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(fv[:,0],fv[:,1],fv[:,2],'bo')