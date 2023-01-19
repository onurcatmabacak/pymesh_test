import pymesh_utils
import numpy as np


filename = "bunny.obj"

mesh = pymesh_utils.load_mesh(filename)

#get_boundary_face_indices = pymesh_utils.get_boundary_face_indices(mesh)
#print(get_boundary_face_indices)
signed_volume = pymesh_utils.get_signed_volume(mesh)
print(signed_volume)