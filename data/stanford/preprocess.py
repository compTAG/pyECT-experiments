import trimesh

mesh = trimesh.load("Armadillo.ply")
mesh.export("../armadillo.obj")
