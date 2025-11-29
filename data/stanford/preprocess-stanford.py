import trimesh

mesh_arm = trimesh.load("Armadillo.ply")
mesh_arm.export("../armadillo.obj")

mesh_bunny = trimesh.load("Bunny.ply")
mesh_bunny.export("../bunny.obj")