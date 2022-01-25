#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:48:20 2022

@author: qcao
"""

import pyvista as pv

mesh = pv.PolyData(v.vertices, padPolyData(v.faces))
mesh.cell_arrays["strains"] = strainsShell4Face

# , strainsShell4Face

voxelSize = np.array((50e-6,)*3)*5 # 50 microns, in meters
voxelsXYZ = (np.array(Sxyz)/voxelSize).astype(int)

grid = pv.UniformGrid()
grid.origin = -np.array(Sxyz)/2 *verticesScaling # (0, 0, 0)
grid.spacing = np.array(voxelSize)*verticesScaling
grid.dimensions = voxelsXYZ

interp = grid.interpolate(mesh, radius=voxelSize[0]*5, sharpness=voxelSize[0], strategy='mask_points')

#%% Plot

movingVertices = pv.PolyData(v.vertices[verticesForce,:])
fixedVertices = pv.PolyData(v.vertices[verticesFixed,:])

rods = pv.PolyData(v.vertices)
rods.lines = padPolyData(v.edges)

plotter = pv.Plotter(shape=(1, 2), border=False, window_size=(2400, 1500))#, off_screen=True)
plotter.background_color = 'k'
plotter.enable_anti_aliasing()

plotter.subplot(0, 0)
plotter.add_text("Voronoi Skeleton (Rods and Plates)", font_size=24)
plotter.add_mesh(rods, show_edges=True)
plotter.add_mesh(movingVertices,color='y',point_size=10)
plotter.add_mesh(fixedVertices,color='c',point_size=10)

plotter.subplot(0, 1)
plotter.add_text("Volumetric Mesh", font_size=24)
plotter.add_mesh(movingVertices,color='y',point_size=10)
plotter.add_mesh(fixedVertices,color='c',point_size=10)

plotter.add_volume(interp)

plotter.link_views()

plotter.camera_position = [(-15983.347882469203, -25410.916652156728, 9216.573794734646),
 (0.0, 0.0, 0.0),
 (0.16876817270434966, 0.24053571467115548, 0.9558555716475535)]

plotter.show()

# print(f"saving to f{imgName}...")
# plotter.show(screenshot=f'{imgName}')