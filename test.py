import pymesh as pm

def load_mesh(filename):

    return pm.load_mesh(filename)

def save_mesh(filename):

    mesh = load_mesh(filename) 
    pm.save_mesh("bunny_saved.obj", mesh)


filename = "bunny.obj"
save_mesh(filename)
