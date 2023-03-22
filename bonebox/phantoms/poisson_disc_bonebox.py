# Poisson disc sampling in arbitrary dimensions via Bridson algorithm
# Implementation by Pavel Zun, pavel.zun@gmail.com
# BSD licence - https://github.com/diregoblin/poisson_disc_sampling
#
# 2022/10/17 Added support for random number generators

# -----------------------------------------------------------------------------
# Based on 2D sampling by Nicolas P. Rougier - https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import numpy as np
from scipy.special import gammainc

# Uniform sampling in a hyperspere
# Based on Matlab implementation by Roger Stafford
# Can be optimized for Bridson algorithm by excluding all points within the r/2 sphere
def hypersphere_volume_sample(center,radius,rng,k=1):
    ndim = center.size
    x = rng.normal(size=(k, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = radius*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(k,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p


# Uniform sampling on the sphere's surface
def hypersphere_surface_sample(center,radius,rng,k=1):
    ndim = center.size
    vec = rng.standard_normal(size=(k, ndim))
    vec /= np.linalg.norm(vec, axis=1)[:,None]
    p = center + np.multiply(vec, radius)
    return p


def squared_distance(p0, p1):
    return np.sum(np.square(p0-p1))

def Bridson_sampling(dims=np.array([1.0,1.0]), radius=0.05, k=30, hypersphere_sample=hypersphere_volume_sample, seed=None, init=None, return_init=False):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    #
    # 10/21/2021 Added init: start with initial seed points instead of a random point (list of tuples/np.arrays)
    #                  return_init: returns initial seed points 
    
    rng = np.random.default_rng(seed=seed)

    ndim=np.array(dims).size

    # size of the sphere from which the samples are drawn relative to the size of a disc (radius)
    sample_factor = 2
    if hypersphere_sample == hypersphere_volume_sample:
        sample_factor = 2
        
    # for the surface sampler, all new points are almost exactly 1 radius away from at least one existing sample
    # eps to avoid rejection
    if hypersphere_sample == hypersphere_surface_sample:
        eps = 0.001
        sample_factor = 1 + eps
    
    def in_limits(p):
        return np.all(np.zeros(ndim) <= p) and np.all(p < dims)

    # Check if there are samples closer than "squared_radius" to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)
        
        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True
        a = []
        for i in range(ndim):
            a.append(slice(indmin[i], indmax[i]))
        with np.errstate(invalid='ignore'):
            if np.any(np.sum(np.square(p - P[tuple(a)]), axis=ndim) < squared_radius):
                return True

    def add_point(p):
        points.append(p)
        indices = (p/cellsize).astype(int)
        P[tuple(indices)] = p

    cellsize = radius/np.sqrt(ndim)
    gridsize = (np.ceil(dims/cellsize)).astype(int)

    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius

    # Positions of cells
    P = np.empty(np.append(gridsize, ndim), dtype=np.float32) #n-dim value for each grid cell
    # Initialise empty cells with NaNs
    P.fill(np.nan)

    points = []
    
    if init is None: # default, start with one random seed point
        add_point(rng.uniform(np.zeros(ndim), dims))
    else: # an initial list of points are specified
        for ip, init_point in enumerate(init):
            add_point(init_point) # add to points
    
    while len(points):
        i = rng.integers(len(points))
        p = points[i]
        del points[i]
        Q = hypersphere_sample(np.array(p), radius * sample_factor, rng, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    
    # return vertices NOT included in the initial set, set elements in P back into np.nan
    if init is not None:

        if return_init==False:
            
            for ip, init_point in enumerate(init): # remove initial points from P
                init_point = np.array(init_point)
                indices = (init_point/cellsize).astype(int) # grid cells with init_points, multiple initial points could share
                P[tuple(indices)] = np.nan # set initial indices of P to nan
        
        else:
            
            init_mask = np.zeros(gridsize, dtype=bool)
            
            for ip, init_point in enumerate(init): # remove initial points from P
                init_point = np.array(init_point)
                indices = (init_point/cellsize).astype(int) # grid cells with init_points, multiple initial points could share 
                init_mask[tuple(indices)] = True
                
            return P[~np.isnan(P).any(axis=ndim)], init_mask[~np.isnan(P).any(axis=ndim)]
        
    return P[~np.isnan(P).any(axis=ndim)]
