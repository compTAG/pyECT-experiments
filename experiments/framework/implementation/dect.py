import torch
import timeit
from pyect import (
    sample_directions_2d,
    WECT,
    weighted_freudenthal
)
from dect.ect import compute_ect
from dect.ect_fn import indicator


torch.set_grad_enabled(False)
device = torch.device("cpu")

img_sizes = [(30, 30), (100, 100), (300, 300), (500, 500), (800, 800)]
num_heights = 1000
num_directions = 50
directions = sample_directions_2d(num_directions, device=device)
transposed_directions = directions.T.contiguous()
n_tests = 20

wect = WECT(directions, num_heights).eval()

for img_size in img_sizes:

    print(f"Testing on {img_size} images:")

    radius = ((0.5 * img_size[0]) ** 2.0 + (0.5 * img_size[1]) ** 2.0) ** .5

    # Building the test set
    img_complexes_pyect = []
    img_complexes_dect = []

    for i in range(n_tests):
        img_complex = weighted_freudenthal(torch.rand(img_size), device=device)
        img_complexes_pyect.append(img_complex)

        x = img_complex.dimensions[0][0]
        edges = img_complex.dimensions[1][0]
        triangles = img_complex.dimensions[2][0]
        simplices = (edges, triangles)
        img_complexes_dect.append((x, simplices))

    ##### PyECT #####

    # Warmup
    for i in range(n_tests):
        output = wect(img_complexes_pyect[i])

    # Test
    start = timeit.default_timer()
    for i in range(n_tests):
        uncompiled_output = wect(img_complexes_pyect[i])
    end = timeit.default_timer()

    print(f"PyECT avg time: {(end - start)/n_tests}")

    ##### DECT #####

    # Warmup
    for i in range(n_tests):
        x, simplices = img_complexes_dect[i]
        output = compute_ect(
        x, 
        v=transposed_directions,
        radius=radius,
        resolution=1000,
        scale=1,
        ect_fn=indicator
    )


    # Test
    start = timeit.default_timer()
    for i in range(n_tests):
        x, simplices = img_complexes_dect[i]

        output = compute_ect(
        x, 
        v=transposed_directions,
        radius=radius,
        resolution=1000,
        scale=1,
        ect_fn=indicator
    )
    end = timeit.default_timer()

    print(f"DECT avg time: {(end - start)/n_tests}")
    print()