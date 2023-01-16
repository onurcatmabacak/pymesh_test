import pymesh as pm
import numpy as np

def load_mesh(filename):

    return pm.load_mesh(filename)

def save_mesh(filename):

    mesh = load_mesh(filename) 
    pm.save_mesh("bunny_saved.obj", mesh)

def tetgen(filename):

    mesh = load_mesh(filename) 
    print(dir(mesh))
    tetgen = pm.tetgen()
    print("tetgen initiliazed")
    tetgen.points = mesh.vertices # Input points.
    print("tetgen initiliazed 1")
    tetgen.triangles = mesh.faces # Input triangles
    print("tetgen initiliazed 2")
    tetgen.max_tet_volume = 0.01
    print("tetgen initiliazed 3")
    tetgen.verbosity = 0
    print("tetgen initiliazed 4")
    tetgen.run() # Execute tetgen
    print("tetgen initiliazed 5")
    mesh = tetgen.mesh

    print(dir(mesh))

def boundary_edges(filename):

    mesh = load_mesh(filename)
    #mesh.enable_connectivity()
    print( mesh.boundary_edges )


def integrate(filename, time_step):

    mesh = load_mesh(filename)
    assembler = pm.Assembler(mesh)
    M = assembler.assemble("mass")

    L = -assembler.assemble("graph_laplacian");
    ##Keep L fixed!
    
    bbox_min, bbox_max = mesh.bbox
    s = np.amax(bbox_max - bbox_min)  # why?
    S = M + (time_step * s) * L
    
    solver = pm.SparseSolver.create("SparseLU")
    solver.compute(S)
    
    mv = M * mesh.vertices

    # print2outershell("before solve")
    vertices = solver.solve(mv)

    # print2outershell("after solve")
    print(vertices, mesh.faces)
    return vertices, mesh.faces

filename = "bunny.obj"
#save_mesh(filename)
#tetgen(filename)

#boundary_edges(filename)
integrate(filename, time_step=0.5)