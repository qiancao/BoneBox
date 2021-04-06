"""
Phantoms.TrabeculaeVoronoi

A voronoi-seeding-based trabecular bone phantom. Refactored version

Authors:  Qian Cao, Xin Xu, Nada Kamona, Qin Li

"""

import numpy as np
from scipy.spatial import Voronoi, Delunay
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
    
    # Extract index of Voronoi vertices within the VOI extent
    Sxyz = np.array(Sxyz)
    ind = np.nonzero(np.prod(np.abs(vor.vertices) < Sxyz/2, axis=1))[0]
    
    return vor, ind

def findUniqueEdges(vor, ind, maxEdgePerFace=8):
    """
    Compute unique edges for each Voronoi cell

    Parameters
    ----------
    vor : scipy.spatial.qhull.Voronoi
        Voronoi tessellation object.
    ind : np.ndarray integers
        Index of vertices within VOI.
    maxEdgePerFace : integer
        Maximum edges per face.

    Returns
    -------
    uniqueEdges, uniqueFaces

    """
    
    # Vertices which have been excluded (outside the extent of VOI)
    ind_exclude = set(np.arange(len(vor.vertices)))-set(np.array(ind))
    
    # Exclude ridges (collection of vertices) containing excluded vertices
    vertices = [vor.ridge_vertices[x] for x in len(vor.ridge_vertices) 
                if vor.ridge_vertices[x]]
    
    # Exclude coplanar vertices, get unique facces and vertices
    
    
    return uniqueEdges, uniqueFaces
    


if __name__ == "__main__":
    # TODO move example driver script into here
    
    print('Running example for TrabeculaeVoronoi')
    Sxyz, Nxyz = (6,6,6), (3,3,3)
    Rxyz = 0.5
    
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=123)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    
    