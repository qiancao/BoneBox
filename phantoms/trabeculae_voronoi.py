"""
phantoms.trabeculae_voronoi

A voronoi-seeding-based trabecular bone phantom. Refactored version.

Authors: Qin Li, Nada Kamona, Xin Xu, Qian Cao

"""

import numpy as np
from scipy.spatial import Voronoi

def makeSeedPointsCartesian(Sxyz, Nxyz):
    """
    Generate perturbed seed points for voronoi tessellation

    Parameters
    ----------
    Sxyz : tuple of floats
        Size of VOI along each dimension (e.g. mm).
    Nxyz : tuple of integers
        Number of points along each dimension.

    Returns
    -------
    points: np.ndarray[N,3]
    
    The VOI is zero-centered.

    """
    
    # Nxyz plus 1, represents number of partitions in each dimension.
    Nxyzp1 = np.array(Nxyz)+1
    
    # Increment along each dimension
    Ixyz = np.array(Sxyz) / Nxyzp1
    
    # 
    
    return points

def perturbSeedPointsCartesian(Sxyz, Nxyz):
    
    
    
    return verts, centroids

if __name__ == "__main__":
    # TODO move example driver script into here
    pass