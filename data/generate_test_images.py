import numpy as np

def generate_test_image(size: int) -> np.ndarray:
    """Generate a grayscale image with a circular region of random intensities."""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 4

    # Create a circular mask
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    # Generate uniform random intensities
    image = np.zeros((size, size), dtype=np.float32)
    image[mask] = np.random.rand(mask.sum()).astype(np.float32)

    return image

def generate_test_images(n: int = 1000, size: int = 128) -> None:
    """Generate n test images and save them safely (no pickling)."""
    images = np.stack([generate_test_image(size) for _ in range(n)], axis=0)
    np.savez_compressed(f"test_images_{size}.npz", images=images)
    print(f"Saved {n} images to test_images_{size}.npz (shape {images.shape})")


if __name__ == "__main__":
    SIZE = 1024 
    NUM_IMGS = 10
    generate_test_images(n=NUM_IMGS, size=SIZE)
