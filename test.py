import pymesh as pm

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

filename = "bunny.obj"
#save_mesh(filename)
tetgen(filename)