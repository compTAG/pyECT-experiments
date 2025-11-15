import numpy as np

def generate_cc_3d(size: int = 100) -> np.ndarray:
    """
    Generate a 3D cubical complex with uniform random weights.

    Returns:
        A (size, size, size) float32 array with values in [0, 1].
    """
    # Uniform random weights on a 3D grid
    return np.random.rand(size, size, size).astype(np.float32)


def generate_cc_3d_images(n: int = 1000, size: int = 100, filename: str = "3D_cc_images.npz") -> None:
    """
    Generate n 3D cubical complexes and save them to an NPZ file.
    The saved array shape will be (n, size, size, size).
    """
    images = np.stack([generate_cc_3d(size) for _ in range(n)], axis=0)
    np.savez_compressed(filename, images=images)
    print(f"Saved {n} cubical complexes to {filename} (shape {images.shape})")


if __name__ == "__main__":
    NUM = 1000
    SIZE = 100
    np.random.seed(42)
    generate_cc_3d_images(n=NUM, size=SIZE, filename="3D_cc_images.npz")
