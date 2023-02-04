import pymesh_utils
import numpy as np
import pymesh as pm 
import pymesh_constants

def tetramesh_generator(mesh):

    tetgen = pymesh_utils.tetgen()
    tetgen.points = mesh.vertices  # Input points.
    tetgen.triangles = mesh.faces  # Input triangles
    tetgen.max_tet_volume = 150
    tetgen.verbosity = 0
    tetgen.merge_coplanar = True
    tetgen.keep_convex_hull = False
    tetgen.run()  # Execute tetgen

    ttrd_mesh = tetgen.mesh

    block = "\n\n\n ############################################################### \n\n\n"

    #print(ttrd_mesh.voxels)
    #print(len(ttrd_mesh.vertices))

    return ttrd_mesh

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

#print( pymesh_utils.integrate(mesh, time_step=0.1, L=1.0) )

#print( pymesh_utils.is_valid_mesh(mesh) )

# mesh.add_attribute(pymesh_constants.FACE_NORMAL)
# normals = mesh.get_face_attribute(pymesh_constants.FACE_NORMAL)
# print( normals )

#print( pm.detect_self_intersection(mesh) )


#mesh.add_attribute(pymesh_constants.FACE_CIRCUMCENTER)

#print(mesh.get_face_attribute(pymesh_constants.FACE_CIRCUMCENTER))

#tetmesh = tetramesh_generator(mesh)
#pymesh_utils.getVoxelDualGraph(tetmesh)
print( len(mesh.vertices) )
mesh, info = pm.collapse_short_edges(mesh, rel_threshold=0.5)

vertices, edges = pm.mesh_to_graph(mesh)
print(len(vertices))
# wire_network = pm.wires.WireNetwork.create_from_data(vertices, edges)
# print( dir(wire_network) )
# print( wire_network.wire_lengths )
# print( np.mean(wire_network.wire_lengths, axis=0) )
# print( np.min(wire_network.wire_lengths) )
# print( len(mesh.vertices) )
# mesh, info = pm.collapse_short_edges(mesh, rel_threshold=0.5)
# print( len(result.vertices) )

wire_network = pm.wires.WireNetwork.create_from_data(vertices, edges)
print(dir(wire_network))
print(len(wire_network.vertices))
inflator = pm.wires.Inflator(wire_network)
inflator.set_profile(8)
inflator.inflate(0.3, per_vertex_thickness=True)
print( dir(inflator) )
mesh = inflator.mesh
pymesh_utils.save_mesh('inflated_mesh.obj', mesh)

print( len(inflator.mesh_vertices) )
print( len(mesh.vertices) )
print( len(inflator.mesh_faces) )
print( len(mesh.faces) )