"""
Phantoms.TrabeculaeVoronoi

A voronoi-seeding-based trabecular bone phantom. Refactored version

Authors:  Qian Cao, Xin Xu, Nada Kamona, Qin Li

"""

import numpy as np
from scipy.spatial import Voronoi
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

    Returns
    -------
    voi : Voronoi object
    
    """
    
    # Number of points
    Np = points.shape[0]
    
    # Generate a random array representing perturbation
    if randState is not None:
        r = np.random.RandomState(randState)
        rarr = r.random((Np,3))
        
    # Sample perturbation from sphere ()
    rarr[:,0] = rarr[:,0]*Rxyz
    Pxyz = utils.Polar2CartesianSphere()
    
    # Perturb points in XY
    
    
    # Voronoi Points in XY, retry on QHull error
    while True:
        try:
            vor = Voronoi(points)
        except:
            raise("QHull Error, retrying")
            continue
        break
    
    return vor

if __name__ == "__main__":
    # TODO move example driver script into here
    
    print('Running example for TrabeculaeVoronoi')
    Sxyz, Nxyz = (6,6,6), (3,3,3)
    Rxy, Rz = 0.5, 0.5
    
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    vor = perturbSeedPointsCartesianUniformXYZ(points, Rxy, Rz)