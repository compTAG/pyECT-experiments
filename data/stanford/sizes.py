from pyect import mesh_to_complex

bunny_cmplx = mesh_to_complex("../bunny.obj", device="cpu")

print("Bunny complex:")
print("vertices:", bunny_cmplx[0][0].shape[0])
print("edges:", bunny_cmplx[1][0].shape[0])
print("faces:", bunny_cmplx[2][0].shape[0])

armadillo_cmplx = mesh_to_complex("../armadillo.obj", device="cpu")

print("\n\nArmadillo complex:")
print("vertices:", armadillo_cmplx[0][0].shape[0])
print("edges:", armadillo_cmplx[1][0].shape[0])
print("faces:", armadillo_cmplx[2][0].shape[0])
