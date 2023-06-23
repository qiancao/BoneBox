# -*- coding: utf-8 -*-
"""

Routines for generating trabecular bone phantoms.

TrabeculaeVoronoi

@author: Qian.Cao

"""

import numpy as np
import scipy
import networkx as nx

import pyvista as pv
import trimesh

def delaunay2simplex(tri, n):
    """
    
    Extract unique edges or faces from scipy.spatial.Delaunay objects.
    https://stackoverflow.com/questions/69512972/how-to-generate-edge-index-after-delaunay-triangulation
    
    Parameters
    ----------
    n
        dimension of the simplex to be extracted. 
        n=2 for edges, 
        n=3 for faces
    
    Returns
    -------
    np.array
    """
    
    from itertools import combinations
    
    simplices = set()
    
    for tri_simplex in tri.simplices:
        for simplex in combinations(tri_simplex,n):
            simplices.add(frozenset(sorted(simplex)))
            
    simplices = [tuple(x) for x in simplices]
    
    return simplices

def removeEdgesWithFaces(edges, faces):
    
    """
    Removes edges that are also contained in faces
    """
    
    if len(faces) > 0:
        from itertools import combinations
        
        edge_set = {frozenset(x) for x in edges} # set of frozen sets
        
        face_edge_set = set()
        for dim in combinations(range(3),2):
            face_edge_set.update({frozenset(x) for x in faces[:,dim]})
            
        edges_pure = np.array([list(x) for x in (edge_set - face_edge_set)])
        
        return edges_pure
        
    else:
        return edges

def generatePoissonScaffold(volume_extent=np.array([5,5,5]),radius=0.5,k=30,seed=None,centered=False, init=None, return_init=False, remove_hull=True):
    """
    Generate Poisson disc sampled points and delunary triangulate for fully-connected faces and edges

    Parameters
    ----------
    volume_extent
        spatial extent of the volume to model
    radius
        minimum radius between points
    k
        see ref. Number of points sampled
    centered
        set origin to middle of volume
    init
        initial seed points, poisson_disc_bonebox does not return the initial seed points, if used as input. 
    return_init
        returns initial seed points in the same order as the poisson disc sampled points.
    
    Notes
    -----
    init_mask does not necessarily contain all points in init in this mode
    
    """
    import poisson_disc_bonebox
    
    if return_init==False: # refactor this later
        vertices_pd = poisson_disc_bonebox.Bridson_sampling(volume_extent, radius=radius, k=k, seed=seed, init=init)
    else:
        vertices_pd , init_mask = poisson_disc_bonebox.Bridson_sampling(volume_extent, radius=radius, k=k, seed=seed, init=init, return_init=True)
    
    # construct init_mask, which are the init vertices
    if init is not None and return_init==False:
        vertices = np.vstack((init, vertices_pd))
        init_mask = np.append(np.ones(len(init),dtype=bool),np.zeros(len(vertices_pd),dtype=bool))
    elif init is not None and return_init==True: # Bridson sampling includes init points
        vertices = vertices_pd
        init_mask = init_mask
    else:
        vertices = vertices_pd
        init_mask = np.zeros(len(vertices_pd),dtype=bool)
    
    if centered:
        vertices = vertices - volume_extent/2
    
    tri = scipy.spatial.Delaunay(vertices)
    edges = delaunay2simplex(tri,2)
    faces = delaunay2simplex(tri,3)
    
    if remove_hull: # remove edges and faces associated with the convex hull
        hull = scipy.spatial.ConvexHull(vertices)
        edges_hull = delaunay2simplex(hull,2)
        faces_hull = delaunay2simplex(hull,3)
        edges = list(set(edges) - set(edges_hull))
        faces = list(set(faces) - set(faces_hull))
        
    edges = np.array(edges)
    faces = np.array(faces)
    
    return vertices, edges, faces, init_mask

def getEdgeLengths(v,e):
    """
    Returns edge lengths from vertices and edges.

    Parameters
    ----------
    v
    e
    """
    
    v = np.array(v)
    e = np.array(e)

    return np.sqrt(np.sum((v[e[:,1]] - v[e[:,0]])**2,axis=1))

def getEdgeDirections(v,e):
    """
    Returns edge lengths from vertices and edges.

    Parameters
    ----------
    v
    e
    """
    
    v = np.array(v)
    e = np.array(e)
    
    l = getEdgeLengths(v,e)

    return (v[e[:,1]] - v[e[:,0]])/l[:,None]

def getNearestPoints(v_from, v_to):
    """
    
    Match nearest point such that v_from[indices] = v_to if v_from[indices] - v_to < max_dist
    
    """
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(v_to)
    distances, indices = nbrs.kneighbors(v_from)
    
    indices = indices.flatten()
    distances = distances.flatten()
    
    return indices, distances

def permute_ef(v_from, v_to, ef_from, max_dist = 0.01):
    """
    
    Permutes edge OR face indices corresponding to v_from to v_to
    
    TODO: implement max_dist.
    
    """
    
    indices, distances = getNearestPoints(v_from, v_to)
    
    indices_from = np.arange(len(v_from))
    
    convert_dict = dict(zip(indices_from, indices))
    
    ef_to = np.vectorize(convert_dict.get)(ef_from) # corresponding edges and faces
    
    return ef_to

def permute_r(v_from, v_to, r_from, max_dist = 0.01):
    """
    
    Permutes radii corresponding to v_from to v_to
    
    same as permute_ef except with applies convert_dictionary to the radii vector.
    
    TODO: this is not right
    
    """
    
    indices, distances = getNearestPoints(v_from, v_to)
    
    indices_from = np.arange(len(v_from))
    
    convert_dict = dict(zip(indices_from, indices))
    
    r_to = np.zeros(v_to.shape[0],dtype=float)
    
    for ind_to, rr_to in enumerate(r_to):
        
        pass
    
    # ef_to = np.vectorize(convert_dict.get)(ef_from) # corresponding edges and faces
    
    return r_to

def ve2g(v,e,weights=None):
    """
    Convert vertex-edges to a networkx graph, edge weights are 1/edge_length

    Parameters
    ----------
    v
    e
    weights : {None, "el", array-like}
        Weights assigned to edges.
        None: all weights are 1. 
        "el": reciprocol of edge lengths. 
        array-like: manually assigned weights.

    Returns
    -------
    G
    """
    
    import networkx as nx
    G = nx.Graph()

    ## add nodes
    node_indices = np.arange(len(v))
    node_attributes = [{'position': v[ind,:]} for ind in node_indices]
    nodes = list(zip(node_indices, node_attributes))
    G.add_nodes_from(nodes)
    
    ## add edge weights
    
    if hasattr(weights, "__len__") and len(weights)==len(e): # user supplied weights
        edges = list(zip(*np.array(e).T,weights))
        G.add_weighted_edges_from(edges)
        
    elif weights=="el":
        el = getEdgeLengths(v,e)
        weights = 1/el
        edges = list(zip(*np.array(e).T,weights))
        G.add_weighted_edges_from(edges)
        
    elif weights==None: # assigns weights as 1
        weights = np.ones(len(e))
        edges = list(zip(*np.array(e).T,weights))
        G.add_weighted_edges_from(edges)
    
    return G

def getAdjDense(G):
    
    A = nx.adjacency_matrix(G).todense()
    
    return A

def getAdjSparse(G):
    
    A = nx.adjacency_matrix(G).todense()
    inds = np.nonzero(A)
    vals = np.array(A[inds]).flatten()
    
    return inds, vals

### Slab Mesh Utilities

def polydata2trimesh(pdmesh):
    """
    
    Converts closed manifold surface mesh from PolyData to trimesh format.

    Parameters
    ----------
    pdmesh

    Returns
    -------
    tmesh

    """
    # https://github.com/pyvista/pyvista/discussions/2268
    faces_as_array = pdmesh.faces.reshape((pdmesh.n_faces, 4))[:, 1:]
    tmesh = trimesh.Trimesh(pdmesh.points, faces_as_array) 
    return tmesh

def trimesh2polydata(tmesh):
    """
    Converts closed manifold surface mesh from trimesh to PolyData format.

    Parameters
    ----------
    tmesh

    Returns
    -------
    pdmesh

    """    
    
    # https://docs.pyvista.org/examples/00-load/wrap-trimesh.html
    pdmesh = pv.wrap(tmesh)
    return pdmesh

def makeSlab(vertices,radii,slab,theta_resolution=10,phi_resolution=10):
    """
    
    Make a slab
    
    slab: index of points in vertices and radii
    # slab mesh https://gist.github.com/flutefreak7/bd621a9a836c8224e92305980ed829b9
    # orient normals outward: https://github.com/BerkeleyAutomation/meshpy/blob/master/meshpy/mesh.py

    Parameters
    ----------
    vertices
    radii
    slab
    theta_resolution : default=10
    phi_resolution : default=10

    Returns
    -------
    poly
    """
    
    import pyvista as pv
    import trimesh
    
    spheres = []
    for s, sind in enumerate(slab):
        sphere = pv.Sphere(radius=radii[sind],center=vertices[sind],
                            theta_resolution=theta_resolution,phi_resolution=phi_resolution)
        spheres.append(polydata2trimesh(sphere))
    
    combined = trimesh.util.concatenate(spheres)
    hull = trimesh.convex.convex_hull(combined)
    
    poly = trimesh2polydata(hull) # normals should point outward: poly.plot_normals(mag=0.2)
    
    assert poly.is_all_triangles, "poly.is_all_triangles is False"
    
    return poly

def pd2volume(pd,volume_extent,ndim,origin=(0,0,0),volume0=None):
    """
    
    Converts a polydata to volume

    Parameters
    ----------
    pd : pyvista.PolyData
        Closed manifold mesh.
    volume_extent : (3,) float tuple of nd.array
        Spatial extent of the volume in x, y and z.
    ndim : (3,) integer tuple of nd.array
        Number of voxels in each dimension.
    volume0
        Initial volume, if specified. Must be of size ndim.

    Returns
    -------
    volume
    """
    
    volume_extent = np.array(volume_extent)
    ndim = np.array(ndim)
    spacing = volume_extent/ndim
    
    # set initial volume
    if volume0 is not None:
        assert tuple(ndim) == volume0.shape, "ndim == volume0.shape failed"
        volume = volume0
    else:
        volume = np.zeros(ndim,dtype=bool)
        
    # point cloud of voxel centers
    bounds = pd.bounds # xmin, xmax, ymin, ymax, zmin, zmax
    
    axes_abs = [] # absolute coordinates for the volume
    axes_bb = [] # axes for the bounding box
    
    for d in range(len(ndim)):
        
        bmin = bounds[2*d]
        bmax = bounds[2*d+1]
        
        axis_ind = np.arange(ndim[d]) + 0.5 # index coordinates
        axis_abs = axis_ind * spacing[d] - origin[d] # absolute coordinates
        axis_bb = (axis_abs > bmin) & (axis_abs < bmax) # coordinates within the bounding box
        
        axes_abs.append(axis_abs)
        axes_bb.append(axis_bb)
        
    # mask for bounding box for pd
    bb = np.meshgrid(*axes_bb)
    bbmask = np.logical_and.reduce([x.flatten() for x in bb])
    
    # extract points
    pts = np.meshgrid(*axes_abs)
    pts = np.array([x.flatten() for x in pts])
    pts = pv.PolyData(pts[:,bbmask].T)
    
    # select points in the inside pd
    selected = pts.select_enclosed_points(pd)
    selected = np.array(selected['SelectedPoints'],dtype=bool)
    
    # assigns values to volume
    volume[bbmask.reshape(volume.shape)] = np.logical_or(volume[bbmask.reshape(volume.shape)],selected)
    
    return volume

def vref2volume(v,r,e,f,volume_extent,ndim,origin=(0,0,0),theta_resolution=5,phi_resolution=5):
    """
    Converts graph (vertices, radii, edges, faces to voxel volume)
    
    Parameters
    ----------
    volume_extent
        volume dimension in mm
    ndim
        number of voxels along each dimension
    origin
        (0,0,0) denotes corner
    
    Returns
    -------
    volume
        voxel volume
    """
    
    e = removeEdgesWithFaces(e,f) # remove edges already in faces
    
    volume = np.zeros(ndim,dtype=bool)
    
    for ee, edge in enumerate(e): # use cleaned edges
        print(f"edges: {ee}/{len(e)}")
        pd = makeSlab(v,r,e[ee],theta_resolution=theta_resolution,phi_resolution=phi_resolution)
        volume = pd2volume(pd,volume_extent,ndim,origin=origin,volume0=volume)
    
    for ff, face in enumerate(f):
        print(f"faces: {ff}/{len(f)}")
        pd = makeSlab(v,r,f[ff],theta_resolution=theta_resolution,phi_resolution=phi_resolution)
        volume = pd2volume(pd,volume_extent,ndim,origin=origin,volume0=volume)
    
    return volume

def volume2surf(volume,voxelSize=(1,1,1),origin=None):
    """
    
    Convenience function, converting volume to surface mesh using routines in MeshUtils
    
    Parameters
    ----------
    volume
        array of 0 and 1. 
        sets boundary voxels of the volume to 0.
    """
    
    volume[0,:,:] = 0
    volume[-1,:,:] = 0
    volume[:,0,:] = 0
    volume[:,-1,:] = 0
    volume[:,:,0] = 0
    volume[:,:,-1] = 0
    
    import MeshUtils
    
    surf_v, surf_f, surf_normals, surf_values = MeshUtils.Voxel2SurfMesh(volume, voxelSize=voxelSize, origin=origin, level=None, step_size=1, allow_degenerate=False)
    surf_v, surf_f = MeshUtils.simplifySurfMeshACVD(surf_v, surf_f, target_fraction=0.5)
    if not MeshUtils.isWatertight(surf_v, surf_f):
        surf_v, surf_f = MeshUtils.repairSurfMesh(surf_v, surf_f)
    assert MeshUtils.isWatertight(surf_v, surf_f), "surf_v and surf_f not watertight"
    
    surf_v[:,(0,1)] = surf_v[:,[1,0]]
    
    return surf_v, surf_f

def Gf_edges(f):
    """
    Generates edges for graph of plates

    Parameters
    ----------
    f : Nx3 array-like
        faces.

    Returns
    -------
    f_e

    """

    f_ind = np.arange(len(f)) # index of faces
    f_e = set() # list of graph edges for faces - this is edges of the augmented graph
    
    for ind, ff in enumerate(f): # ff is [v0,v1,v2]
    
        same_vert = np.zeros(f.shape,dtype=np.int32)
        for d in range(len(ff)):
            same_vert = same_vert + (ff[d]==f)
            
        same_vert_arr = np.sum(same_vert,axis=1) # Nx1 vector representing same vertices     
        neighbors = f_ind[same_vert_arr == 2] # faces which share two vertices with ff
        
        for nn in neighbors:
            
            f_e.add(frozenset([ind,nn]))
    
    f_e = np.array([list(x) for x in f_e])
    
    return f_e
   
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
    
    faceNormals = np.array([computeNormal(x) for x in faceVertices])
    
    return faceNormals

def angle_btw(v0, v1):
    """
    returns angle (in radians) between v0 and v1.
    
    Parameters
    ----------
    v0 : array-like (3,) or (N,3)
        normal vectors corresponding to a face.
    V1 : array-like (3,) or (N,3), must be broadcastable with v0

    Returns
    -------
    None.
    
    Notes
    -----
    This is the minimum angle between two LINES (<=90 degrees), not vectors
 
    """
    
    def normalize(x):
        x = np.array(x)
        x = (x.T/np.linalg.norm(x,axis=-1)).T
        return x
    
    v0, v1 = normalize(v0), normalize(v1)
    
    return np.arccos(np.abs(np.dot(v0,v1)))

def cosine_btw(v0, v1):
    """
    returns angle (in radians) between v0 and v1.
    
    Parameters
    ----------
    v0 : array-like (3,) or (N,3)
        normal vectors corresponding to a face.
    V1 : array-like (3,) or (N,3), must be broadcastable with v0

    Returns
    -------
    None.
    
    Notes
    -----
    This is the minimum angle between two LINES (<=90 degrees), not vectors
 
    """
    
    def normalize(x):
        x = np.array(x)
        x = (x.T/np.linalg.norm(x,axis=-1)).T
        return x
    
    v0, v1 = normalize(v0), normalize(v1)
    
    return np.abs(np.dot(v0,v1))

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
    
    faceCentroids = np.array([computeCentroid(x) for x in faceVertices])
    
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
    
    faceAreas = np.array([computeArea(x) for x in faceVertices])
    
    return faceAreas

def computeFaceEqualateralities(faceVertices):
    
    def computeEqualaterality(verts):
        
        v0 = verts[0,:]
        vk = np.linalg.norm(verts[2:,:] - v0)
        vj = np.linalg.norm(verts[1:-1,:] - v0)
        vl = np.linalg.norm(vk-vj)
        
        CV = np.std([vk,vj,vl]) / np.mean([vk,vj,vl])
        
        return CV
    
    faceEqualaterality = np.array([computeEqualaterality(x) for x in faceVertices])
    
    return faceEqualaterality

# Convenience functions for face properties
def vf2verts(v,f):
    # returns face vertices for use in vf2... functions
    return v[f,:]

def vf2normals(v,f):
    return computeFaceNormals(vf2verts(v,f))

def vf2centroids(v,f):
    return computeFaceCentroids(vf2verts(v,f))

def vf2areas(v,f):
    return computeFaceAreas(vf2verts(v,f))

def vf2equalateralities(v,f):
    return computeFaceEqualateralities(vf2verts(v,f))

def Gf_edge_weights(v,f,fe):
    """
    
    Generate edge weights for plate graph.
    
    Default is cosine.

    Parameters
    ----------
    v : array-like (N,3)
        vertex array.
    f : array-like (M,3)
        face index (to vertex array).
    fe : (K,2)
        edge index (to f).

    Returns
    -------
    weights
        Edge weights corresponding to fe.

    """
    
    weights = np.zeros(len(fe))
    
    for ind, ffe in enumerate(fe):
        
        f0, f1 = f[ffe]
    
        n0, n1 = computeFaceNormals(v[f0[None,:]]).flatten(), computeFaceNormals(v[f1[None,:]]).flatten()
        
        weights[ind] = cosine_btw(n0,n1)
        
    return weights
