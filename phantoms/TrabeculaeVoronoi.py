"""
Phantoms.TrabeculaeVoronoi

A voronoi-seeding-based trabecular bone phantom.

Refactored April 2021.

Authors:  Qian Cao, Xin Xu, Nada Kamona, Qin Li

"""

import numpy as np
# Delunary to test for colinearlity
from scipy.spatial import Voronoi, Delaunay
# import optimesh # May be a good idea for CVT post perturbation
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
        DESCRIPTION.

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
       return [list(x) for x in set(tuple(x) for x in data_list)]
        
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
    # This may not be necessary
    
    # Extract edges from faces
    ridge_circ = [vor.ridge_vertices[x] + [vor.ridge_vertices[x][0]] 
                  for x in ridge_ind] # This is a copy of ridge
    
    
    # Apply uniqueness to faces and edges
    uniqueFaces = uniq(ridge)
    uniqueEdges = []
    
    return uniqueEdges, uniqueFaces


if __name__ == "__main__":
    
    print('Running example for TrabeculaeVoronoi')
    Sxyz, Nxyz = (10,10,10), (5,5,5)
    Rxyz = 0.5
    
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=123)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    uniq_edges, uniq_faces = findUniqueEdgesAndFaces(vor, ind)
    
    