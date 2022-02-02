# BoneBox

Tools for bone modeling, evaluation and biomarker development.

### Example: Generating a simple Voronoi-based Trabecular Model

```
import numpy as np
from bonebox.Phantoms.TrabeculaeVoronoi import *

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
```
