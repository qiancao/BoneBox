"""
Routines for meshing originally in FEA/fea.py

Does NOT depend on pychrono

Qian Cao 2022.09.04

"""

import numpy as np
import nrrd

from skimage import measure
import tetgen # tetrahedralization
import trimesh # general mesh ops
import pyvista as pv

def addPlatten(volume, plattenThicknessVoxels, plattenValue=None, airValue=None, trimVoxels=0):
    # adds compression plates in Z
    # leaves a single-voxel space at the edge of volume (for isosurface)
    
    vmax = np.max(volume)
    vmin = np.min(volume)
    
    if plattenValue == None:
        plattenValue = vmax
        
    if airValue == None:
        airValue = vmin
    
    # Leaves single-voxel space at edge of volume for isosurface ops
    volume[:,:,0] = airValue
    volume[:,:,-1] = airValue
    
    # Define platten
    volume[(1+trimVoxels):(-1-trimVoxels),(1+trimVoxels):(-1-trimVoxels),1:plattenThicknessVoxels] = plattenValue
    volume[(1+trimVoxels):(-1-trimVoxels),(1+trimVoxels):(-1-trimVoxels),-plattenThicknessVoxels:-1] = plattenValue

    return volume

def set_volume_bounds(volume, airValue=None, bounds = 1):
    # set boundaries of volume to airValue
    
    if airValue is None:
        airValue = np.min(volume)
        
    volume[:(bounds+1),:,:] = airValue
    volume[-(bounds+1):,:,:] = airValue
    volume[:,:(bounds+1),:] = airValue
    volume[:,-(bounds+1):,:] = airValue
    volume[:,:,:(bounds+1)] = airValue
    volume[:,:,-(bounds+1):] = airValue
    
    return volume

def filter_connected_volume(volume):
    # filter out components unconnected to the main bone structure
    # performs connected component analysis and preserves only the largest connected component
    
    labels = measure.label(volume,connectivity=1)
    values = np.unique(labels) # get all labels
    values.sort()
    values = values[1:] # discard zeros (background)
    num_voxels = [np.sum(labels==x) for x in values]
    largest_component_label = values[np.argmax(num_voxels)]
    
    vmin = np.min(volume)
    vmax = np.max(volume)
    
    volume_out = np.ones(volume.shape,dtype=volume.dtype) * vmin
    volume_out[labels==largest_component_label] = vmax
    
    return volume_out

def filter_connected_mesh(faces):
    # filter out components unconnected to the main bone structure
    pass

def Voxel2HexaMeshIndexCoord(volume):
    """
    Directly convert voxels to hexamesh (bricks) and returns mesh in index coordinates
        
    Example: to retrieve nodes corresponding to element 217:
    nodesSortedUnique[elements[217],:]
    
    Given the default voxelSize and origin, coordinates range from (-0.5 to dimXYZ+0.5)
    
    nodesSortedUnique.shape = (nodes,3)
    """
    
    xx, yy, zz = np.nonzero(volume)
    nElements = len(xx)
    
    # Compute list of vertex (node) coordinates
    nodes = np.empty((3,nElements,8)) #(xyz, N, verts)
    nodes[:] = np.NaN
    
    nodes[:,:,0] = np.vstack((xx-0.5, yy-0.5, zz-0.5))
    nodes[:,:,1] = np.vstack((xx+0.5, yy-0.5, zz-0.5))
    nodes[:,:,2] = np.vstack((xx+0.5, yy+0.5, zz-0.5))
    nodes[:,:,3] = np.vstack((xx-0.5, yy+0.5, zz-0.5))
    nodes[:,:,4] = np.vstack((xx-0.5, yy-0.5, zz+0.5))
    nodes[:,:,5] = np.vstack((xx+0.5, yy-0.5, zz+0.5))
    nodes[:,:,6] = np.vstack((xx+0.5, yy+0.5, zz+0.5))
    nodes[:,:,7] = np.vstack((xx-0.5, yy+0.5, zz+0.5))
    
    nodes = np.reshape(nodes, (3,-1))
    
    # Simplify, keep only unique nodes (faster than np.unique)
    nodesSortedInds = np.lexsort(nodes) # last element (z) first
    nodesSorted = nodes[:,nodesSortedInds]
    dnodes = np.sum(np.diff(nodesSorted, axis=1), axis=0) # assumes nodes are sorted lexigraphically
    mask = np.hstack((True, dnodes!=0)) # mask array with unique nodes set to True on first appearance
    nodesSortedUnique = nodesSorted[:,mask] # *** final list of node coordinates
    
    # Initialize array of elements, indices to nodes in nodesSortedUnique
    elements = np.zeros(nElements*8, dtype=np.int64) # final list of elements (index of nodes in nodeSortedUnique)
    nodeIndsOriginal = np.arange(nElements*8)
    nodeIndsOriginalSorted = nodeIndsOriginal[nodesSortedInds]
    nodeIndsReduced = np.cumsum(mask)-1
    elements[nodeIndsOriginalSorted] = nodeIndsReduced
    elements = np.reshape(elements, (nElements, 8))
    
    nodesSortedUnique = nodesSortedUnique.T
    
    return nodesSortedUnique, elements

def HexaMeshIndexCoord2Voxel(nodes, elements, dim):
    """
    Convert hexamesh (bricks) in index coordinates to volume in voxels
    
    dim: dimension of volume in x, y and z in voxels (tuple)
        
    Example: to retrieve nodes corresponding to element 217:
    nodesSortedUnique[elements[217],:]
    
    Given the default voxelSize and origin, coordinates range from (-0.5 to dimXYZ+0.5)
    
    nodesSortedUnique.shape = (nodes,3)
    
    """
    
    volume = np.zeros(dim, dtype=bool) # initialize volume of False
    xyz = nodes[elements,:][:,0,:] + 0.5 # voxel coordinates of bone
    xyz = xyz.astype(int)
    volume[tuple(xyz.T)] = True
    
    return volume

def HexaMeshIndexCoord2VoxelValue(nodes, elements, dim, elementValues):
    """
    Convert hexamesh (bricks) in index coordinates to volume in voxels with value of voxels assigned according to elementValues.
    
    dim: dimension of volume in x, y and z in voxels (tuple)
    elementValues: len(elements) == len(elementValues)
        
    Example: to retrieve nodes corresponding to element 217:
    nodesSortedUnique[elements[217],:]
    
    Given the default voxelSize and origin, coordinates range from (-0.5 to dimXYZ+0.5)
    
    nodesSortedUnique.shape = (nodes,3)
    
    """
    
    volume = np.zeros(dim, dtype=elementValues.dtype) # initialize volume of False
    xyz = nodes[elements,:][:,0,:] + 0.5 # voxel coordinates of bone
    xyz = xyz.astype(int)
    volume[tuple(xyz.T)] = elementValues
    
    return volume

def Index2AbsCoords(nodes, dim, voxelSize=(1,1,1), origin=(0,0,0)):
    """
    Convert array of node coordinates from index coordinates to absolute coordinates
        
        dim: volume dimension in voxels (tuple)
        origin: shift of origin from center of volume
        (0,0,0) corresponds to center of volume (default), (-X/2, -Y/2, -Z/2) refers to "top left corner".
    """

    dim = np.array(dim)
    voxelSize = np.array(voxelSize)
    origin = np.array(origin)
    nodes = (nodes - dim/2) * voxelSize + origin
    return nodes

def Abs2IndexCoords(nodes, dim, voxelSize=(1,1,1), origin=(0,0,0)):
    """
    Convert array of node coordinates from abs coordinates to index coordinates
        
        dim: volume dimension in voxels (tuple)
        origin: shift of origin from center of volume
        (0,0,0) corresponds to center of volume (default), (-X/2, -Y/2, -Z/2) refers to "top left corner".
    """

    dim = np.array(dim)
    voxelSize = np.array(voxelSize)
    origin = np.array(origin)
    nodes = (nodes - origin) / voxelSize + dim/2
    return nodes

def Voxel2SurfMesh(volume, voxelSize=(1,1,1), origin=None, level=None, step_size=1, allow_degenerate=False):
    # Convert voxel image to surface
    
    if level == None:
        level = (np.max(volume))/2
    
    # vertices, faces, normals, values = \
    #     measure.marching_cubes_lewiner(volume = volume, level = level, spacing = voxelSize, \
    #                                    step_size = step_size, allow_degenerate = allow_degenerate)
    vertices, faces, normals, values = \
        measure.marching_cubes(volume = volume, level = level, spacing = voxelSize, \
                               step_size = step_size, allow_degenerate = allow_degenerate)
            
    return vertices, faces, normals, values

def Surf2TetMesh(vertices, faces, order=1, verbose=1, **tetkwargs):
    # Convert surface mesh to tetrahedra
    # https://github.com/pyvista/tetgen/blob/master/tetgen/pytetgen.py
    
    tet = tetgen.TetGen(vertices,faces)
    tet.tetrahedralize(order=order, verbose=verbose, **tetkwargs)
    
    nodes = tet.node
    elements = tet.elem
    
    return nodes, elements, tet

def smoothSurfMesh(vertices, faces, **trimeshkwargs):
    # smooths surface mesh using Mutable Diffusion Laplacian method
    
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    trimesh.smoothing.filter_mut_dif_laplacian(mesh, **trimeshkwargs)
    
    return mesh.vertices, mesh.faces

# TODO: FQMR doesn't seem to generate watertight meshes
def simplifySurfMeshFQMR(vertices, faces, target_fraction=0.25, lossless=True,
                      preserve_border=True, **kwargs):
    # simplify surface mesh
    import pyfqmr
    
    # if target_count is not specified
    Nfaces = faces.shape[0]
    if "target_count" not in kwargs.keys():
        kwargs["target_count"] = round(Nfaces*target_fraction)
    
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(vertices, faces)
    mesh_simplifier.simplify_mesh(**kwargs)
    
    vertices, faces, normals = mesh_simplifier.getMesh()
    
    return vertices, faces

def simplifySurfMeshOpen3D(vertices, faces, target_fraction=0.25, lossless=True,
                      preserve_border=True, **kwargs):
    # simplify surface mesh, use open3D, does not always generate watertight mesh
    import open3d as o3d # mesh simplification TODO: Optional
    
    Nfaces = faces.shape[0]
    target_count = round(Nfaces*target_fraction)

    mesh = trimesh.Trimesh(vertices = vertices, faces = faces).as_open3d
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_count)

    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

def simplifySurfMeshACVD(vertices, faces, target_fraction=0.25):
    # simplify surface mesh, use pyacvd
    import pyacvd
    
    Nfaces = faces.shape[0]
    target_count = round(Nfaces*target_fraction)
    
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    mesh = pv.wrap(mesh)
    
    clus = pyacvd.Clustering(mesh)
    clus.cluster(target_count)
    
    mesh = clus.create_mesh()
    
    # https://github.com/pyvista/pyvista/discussions/2268
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    tmesh = trimesh.Trimesh(vertices = mesh.points, faces = faces_as_array)
    
    return tmesh.vertices, tmesh.faces

def repairSurfMesh(vertices, faces):
    import pymeshfix
    vclean, fclean = pymeshfix.clean_from_arrays(vertices, faces)
    return vclean, fclean

def isWatertight(vertices, faces):
    # Check if mesh is watertight
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    return mesh.is_watertight

def saveSurfaceMesh(filename, vertices, faces):
    # Save surface mesh (tested on .STL only)
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
    mesh.export(filename)

def saveTetrahedralMesh(filename, tet):
    pv.save_meshio(filename, tet.grid)

def cropCubeFromCenter(img,length):
    
    x0,y0,z0 = np.array(img.shape)//2
    R = length//2
    
    return img[slice(x0-R,x0+R+1),
               slice(y0-R,y0+R+1),
               slice(z0-R,z0+R+1)]

if __name__ == "__main__":
    pass