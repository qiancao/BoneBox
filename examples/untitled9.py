# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:10:09 2022
@author: Qian.Cao

* install vtk==9.1.0 or else windows won't close

"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

grid = pv.UniformGrid(
    dims=(100, 100, 100),
    spacing=(1, 1, 1),
    origin=(-50, -50, -50),
)

vertices = np.array([[-40,-40,-40],[20,20,20],[0,0,-30]]).astype(float)
cells = np.array([[3,0,1,2]])

mesh = pv.PolyData(vertices, cells)
mesh.cell_data["values"] = 1.2

# # sample = grid.sample(mesh)
grid_dist = grid.compute_implicit_distance(mesh)

dargs = dict(cmap="coolwarm", clim=[0,1], scalars="values")

p = pv.Plotter()
p.add_mesh(grid.outline(), color='k')
p.add_mesh(mesh, render_points_as_spheres=True, **dargs)
p.show()

dist = grid_dist.point_data['implicit_distance']
dist = np.array(dist).reshape((100, 100, 100))

plt.imshow(dist[:,50,:])