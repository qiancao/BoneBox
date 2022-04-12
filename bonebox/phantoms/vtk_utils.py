#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:20:28 2022

@author: qcao

https://vtk.org/pipermail/vtkusers/2007-December/044165.html

https://python.hotexamples.com/examples/vtk/-/vtkPolyDataToImageStencil/python-vtkpolydatatoimagestencil-function-examples.html

https://www.cb.uu.se/~aht/Vis2014/lecture2.pdf

https://github.com/bpinsard/misc/blob/master/surf_fill.py

https://discourse.vtk.org/t/slow-voxelisation-with-vtkpolydatatoimagestencil/5571

https://kitware.github.io/vtk-examples/site/Cxx/PolyData/PolyDataToImageData/

https://github.com/DlutMedimgGroup/AnatomySketch-Software/blob/7c2831c8b369892d2344415ad6d66fffc3333e75/AID/python/pylib/improcessing/Convert_Polydata_To_Imagedata.py

# based on test_vtk_rasterize

https://vtk.org/doc/release/5.6/html/a01461.html

https://vtk.org/pipermail/vtkusers/2012-July/075442.html

https://public.kitware.com/pipermail/vtkusers/2014-September/085140.html

https://public.kitware.com/pipermail/vtkusers/2002-April/011020.html

https://python.hotexamples.com/examples/vtk/-/vtkLinearExtrusionFilter/python-vtklinearextrusionfilter-function-examples.html

make_patterned_polydata

"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import sys
sys.path.append("../phantoms/") # TODO: remove this when incorporated into library
sys.path.append("../FEA/") # from examples folder

from TrabeculaeVoronoi import *
# from fea import *

import vtk
from vtk.util import numpy_support

import edt

def vtkPolyData(vertices, faces, edges):
    
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(vertices))
    for i, pt in enumerate(vertices):
        points.InsertPoint(i, pt)
        
    # vtkCellArray - faces
    tris  = vtk.vtkCellArray()
    for vert in faces:
        tris.InsertNextCell(len(vert))
        for v in vert:
            tris.InsertCellPoint(v)
            
    # vtkCellArray - edge
    tris  = vtk.vtkCellArray()
    for vert in edges:
        tris.InsertNextCell(len(vert))
        for v in vert:
            tris.InsertCellPoint(v)
            
    # vtkPolyData
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(tris)
    
    return pd

def voxelize_surface(vertices, polys, dims, spacing, origin=None, value=1):
    """
    Surface rasterization in 3D using VTK
    
    input vertices are in VTK ARRAY COORDINATES!!! Must be integers.
    dims is the shape of image
    
    """
    
    if origin is None:
        origin = -np.array(dims)*np.array(spacing)/2
    
    # vtkPoints
    # vertices = convertAbs2Array(vertices, spacing, dims) # CONVERT to array coordinates
    
    # spacing = (1,1,1)
    # origin = (0,0,0)
    
    # print(vertices)
    
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(vertices))
    for i, pt in enumerate(vertices):
        points.InsertPoint(i, pt)
        
    # vtkCellArray
    tris  = vtk.vtkCellArray() # triangles or any polydata?
    for vert in polys:
        tris.InsertNextCell(len(vert))
        for v in vert:
            tris.InsertCellPoint(v)

    # vtkPolyData
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(tris)
    del points, tris

    # vtkImageData
    whiteimg = vtk.vtkImageData()
    whiteimg.SetDimensions(dims)
    whiteimg.SetSpacing(spacing)
    whiteimg.SetOrigin(origin) # Does not seem to be setting the origin correctly

    # vtkInformation -> SetPointDataActiveScalarInfo
    info = vtk.vtkInformation()
    whiteimg.SetPointDataActiveScalarInfo(info, vtk.VTK_UNSIGNED_CHAR, value)
    
    # PointData -> SetScalars
    ones = np.ones(np.prod(dims),dtype=np.uint8)
    whiteimg.GetPointData().SetScalars(numpy_support.numpy_to_vtk(ones))
    
    # vtkPolyDataToImageStencil
    pdtis = vtk.vtkPolyDataToImageStencil()
    pdtis.SetInputData(pd)
    pdtis.SetOutputSpacing(whiteimg.GetSpacing())
    pdtis.SetOutputOrigin(whiteimg.GetOrigin())
    pdtis.SetOutputWholeExtent(whiteimg.GetExtent())
    pdtis.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteimg)
    imgstenc.SetStencilConnection(pdtis.GetOutputPort())
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()
    
    data = numpy_support.vtk_to_numpy(
        imgstenc.GetOutput().GetPointData().GetScalars()).reshape(dims).transpose(2,1,0)
    
    # verticesArray = convertAbs2Array(vertices[polys,:], spacing, dims).round().astype(int) # vertices already in array coordinates
    # for ind, v in enumerate(vertices.round().astype(int)):
    #     data[v[0],v[1],v[2]] = value
    
    del pd,vertices,whiteimg,pdtis,imgstenc
    return data

def voxelize_extruded_surface(vertices, polys, dims, spacing, extruded_thickness=None, origin=None, value=1):
    """
    
    Surface rasterization in 3D using VTK
    
    input vertices are in VTK world COORDINATES!
    dims is the shape of image
    
    """
    
    if origin is None:
        origin = -np.array(dims)*np.array(spacing)/2
        
    if extruded_thickness is None: # default extrude by half voxel
        extruded_thickness = spacing[0]
    face_normal = computeFaceNormals([vertices])[0]
    extrudeVector = face_normal * extruded_thickness
    
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(vertices))
    for i, pt in enumerate(vertices):
        points.InsertPoint(i, pt)
        
    # vtkCellArray
    tris  = vtk.vtkCellArray() # triangles or any polydata?
    for vert in polys:
        tris.InsertNextCell(len(vert))
        for v in vert:
            tris.InsertCellPoint(v)

    # vtkPolyData
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(tris)
    del points, tris
    
    # extrude polydata along its normal
    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.CappingOn()
    extrude.SetInputData(pd)
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(*extrudeVector)
    extrude.Update()

    # vtkImageData
    whiteimg = vtk.vtkImageData()
    whiteimg.SetDimensions(dims)
    whiteimg.SetSpacing(spacing)
    whiteimg.SetOrigin(origin) # Does not seem to be setting the origin correctly

    # vtkInformation -> SetPointDataActiveScalarInfo
    info = vtk.vtkInformation()
    whiteimg.SetPointDataActiveScalarInfo(info, vtk.VTK_UNSIGNED_CHAR, value)
    
    # PointData -> SetScalars
    ones = np.ones(np.prod(dims),dtype=np.uint8)
    whiteimg.GetPointData().SetScalars(numpy_support.numpy_to_vtk(ones))
    
    # vtkPolyDataToImageStencil
    pdtis = vtk.vtkPolyDataToImageStencil()
    pdtis.SetInputData(extrude.GetOutput()) # changed from pd to extrude
    pdtis.SetOutputSpacing(whiteimg.GetSpacing())
    pdtis.SetOutputOrigin(whiteimg.GetOrigin())
    pdtis.SetOutputWholeExtent(whiteimg.GetExtent())
    pdtis.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteimg)
    imgstenc.SetStencilConnection(pdtis.GetOutputPort())
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()
    
    data = numpy_support.vtk_to_numpy(
        imgstenc.GetOutput().GetPointData().GetScalars()).reshape(dims).transpose(2,1,0)
    
    del pd,vertices,whiteimg,pdtis,imgstenc
    return data

def voxelize_plates(vertices, faces, dims, spacing, values):
    
    # input vertices are in world coordinates, convert to integer coordinates before input to vtk ** TODO: This may not be necessary
    # vertices = ((convertAbs2Array(vertices, spacing, dims)+0.5) - np.array(dims)/2).round().astype(int)
    # spacing = (1, 1, 1)
    
    volume = np.zeros(dims,dtype=float)
    
    for ind, face in enumerate(faces):
        
        # vertices are converted to zero-centered, spacing=1, coordinate system (wish I didn't have to do that, check vtk documentation)
        # offset of array coordinates by 0.5
        dv = voxelize_extruded_surface(vertices[face,:], [[0,1,2],], dims, spacing, extruded_thickness=spacing[0]*2, origin=None, value=1)
        dv = dv * values[ind]
        
        # print(vertices[face,:])
        print(f"face: {ind}/{len(faces)}, max val: {np.max(dv)}, number voxels: {np.sum(dv>0)}, area: {computeFaceAreas([convertAbs2Array(vertices[face,:], spacing, dims)])}")
        
        volume += dv
        
    return volume

def voxelize_rods(vertices, edges, dims, spacing, values):
    
    volume = np.zeros(dims,dtype=float)
    
    for ind, edge in enumerate(edges):
        dv = makeSkeletonVolumeEdges(vertices[edge,:], [[0,1],], spacing, dims)
        volume += dv
        
    return volume

def plot_mesh(volume, dims, spacing):
    
    vertices, faces, normals, values = Voxel2SurfMesh(volume, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
    tmesh = trimesh.Trimesh(vertices, faces=faces)
    surf = pv.wrap(tmesh)
    
    pl = pv.Plotter()
    pl.add_mesh(surf)
    return pl

if __name__ == "__main__":
    
    plt.close("all")

    dims = (500,500,500) # volumeSizeVoxels
    spacing = (1,1,1) # voxelSize
    
    # triangular mesh ()
    vertices = np.array([[0,50,-50],[0,50,50],[0,0,20]]).astype(float) # must be integer coordinates!!!
    face = np.array([[0,1,2],])
    # volume0 = voxelize_plates(vertices, face, dims, spacing, [2,])
    volume0 = voxelize_surface(vertices, [[0,1,2],], dims, spacing, origin=None, value=1)
    
    vertices = np.array([[0,-120,-120],[0,120,120],[120,120,0]]).astype(float)
    line = np.array([[0,1],])
    # volume1 = voxelize_line(vertices, line, dims, spacing)
    volume1 = voxelize_rods(vertices, line, dims, spacing, [2,])
    
    volume = volume0 + volume1
    
    fig = plt.figure()
    plt.imshow(volume[250,:,:])
    
    #%% Plates
    
    dims = (500,500,500) # volumeSizeVoxels
    spacing = (1,1,1) # voxelSize
    
    # triangular mesh ()
    vertices = np.array([[0,50,-50],[0,50,50],[0,0,20]]).astype(float) # must be integer coordinates!!!
    face = np.array([[0,1,2],])
    volume0 = voxelize_plates(vertices, face, dims, spacing, [2,])
    # volume0 = voxelize_surface(vertices, [[0,1,2],], dims, spacing, origin=None, value=1)
    
    vertices = np.array([[0,-120,-120],[0,120,120],[120,120,0]]).astype(float)
    line = np.array([[0,1],])
    # volume1 = voxelize_line(vertices, line, dims, spacing)
    volume1 = voxelize_rods(vertices, line, dims, spacing, [2,])
    
    volume = volume0 + volume1
    
    fig = plt.figure()
    plt.imshow(volume[250,:,:])
    
    vertices, faces, normals, values = Voxel2SurfMesh(volume, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
    tmesh = trimesh.Trimesh(vertices, faces=faces)
    surf = pv.wrap(tmesh)
    surf.plot()
    
    #%% This doesn't work
    
    # plt.close("all")
    
    # dims = (200,200,200)
    # spacing = np.array([5.e-05, 5.e-05, 5.e-05])
    # # spacing = np.array([1, 1, 1])
    
    # vertices = np.array([[-2.39289471e-03, -5.71013001e-05,  2.71962678e-03],
    #                       [-2.39517461e-03, -2.27506353e-03,  2.50233336e-03],
    #                       [-9.62778150e-04, -9.40870800e-04,  2.77942200e-03]])
    
    # face = [[0,1,2],]
    
    # verticesVTK = (convertAbs2Array(vertices, spacing, dims)+0.5 - np.array(dims)/2).round()
    
    # volume = voxelize_plates(vertices, face, dims, spacing, [2,])
    
    # fig = plt.figure()
    # plt.imshow(volume[:,:,152])
    
    # vertices, faces, normals, values = Voxel2SurfMesh(volume, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
    # tmesh = trimesh.Trimesh(vertices, faces=faces)
    # surf = pv.wrap(tmesh)
    # surf.plot()
    
    # #%% This does
    
    # dims = (200,200,200)
    # # # # spacing = np.array([5.e-05, 5.e-05, 5.e-05])
    # spacing = np.array([1, 1, 1])
    
    # # # vertices = np.array([[-2.39289471e-03, -5.71013001e-05,  2.71962678e-03],
    # # #                      [-2.39517461e-03, -2.27506353e-03,  2.50233336e-03],
    # # #                      [-9.62778150e-04, -9.40870800e-04,  2.77942200e-03]])
    
    # vertices2 = np.array([[ 52,  98, 154],
    #                       [ 52,  54, 150],
    #                       [ 80,  81, 155]]) - np.array(dims)/2
    
    # # vertices = verticesVTK # this doesn't
    # vertices = vertices2 # this does work
    # # vertices = vertices3 # this does work
    
    # face = [[0,1,2],]
    
    # # # # verticesArray = convertAbs2Array(vertices[face,:], spacing, dims).round().astype(int)
    
    # volume = voxelize_plates(vertices, face, dims, spacing, [2,])
    
    # fig = plt.figure()
    # plt.imshow(volume[:,:,152])
    
    pass