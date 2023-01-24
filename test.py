import pymesh_utils
import numpy as np
import pymesh

filename = "bunny.obj"

mesh = pymesh_utils.load_mesh(filename)

#get_boundary_face_indices = pymesh_utils.get_boundary_face_indices(mesh)
#print(get_boundary_face_indices)
#signed_volume = pymesh_utils.get_signed_volume(mesh)
#print(signed_volume)

# hull = pymesh.convex_hull(mesh, engine='qhull', with_timing=False)
# print( mesh.get_vertex_adjacent_vertices(1) )
# quit()
# print( pymesh_utils.scaleHull(hull, 2.0) )

print( pymesh_utils.integrate(mesh, time_step=0.1, L=1.0) )