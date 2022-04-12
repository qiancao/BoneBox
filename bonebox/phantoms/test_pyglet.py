#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:23:54 2022

@author: qcao

References:
    
Extract numpy array from pyglet image:
    https://stackoverflow.com/questions/55742197/how-to-change-a-pyglet-image-into-a-numpy-array
    
https://medium.com/@yvanscher/opengl-and-pyglet-basics-1bd9f1721cc6

"""

import numpy as np
import pyglet

# import all of opengl functions
from pyglet.gl import *

win = pyglet.window.Window()

@win.event
def on_draw():
    # create a line context
    glBegin(GL_LINES)
    # create a line, x,y,z
    glVertex3f(100.0,100.0,0.25)
    glVertex3f(200.0,300.0,-0.75)
    glEnd()

pyglet.app.run()