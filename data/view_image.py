import numpy as np
import matplotlib.pyplot as plt
import os
import random
from typing import Optional

def load_and_view_random_image(npz_file_path: str, array_key: str = 'images') -> None:
    """
    Loads an NPZ file, selects a random image from the specified array key, 
    and displays it using Matplotlib.
    
    Args:
        npz_file_path: The path to the .npz file (e.g., 'fashion_mnist_images.npz').
        array_key: The key used when saving the array (default is 'images').
    """
    if not os.path.exists(npz_file_path):
        print(f"Error: The file '{npz_file_path}' was not found.")
        return

    try:
        # 1. Load the NPZ file
        print(f"Loading data from {npz_file_path}...")
        
        # Use np.load() with a context manager for safe and efficient loading
        with np.load(npz_file_path, allow_pickle=True) as data:
            if array_key not in data:
                print(f"Error: Array key '{array_key}' not found in the NPZ file.")
                print(f"Available keys are: {list(data.keys())}")
                return

            # Retrieve the images array
            images = data[array_key]
            
            if images.ndim < 2:
                print(f"Error: The array '{array_key}' has an unexpected shape: {images.shape}. Expected at least 2 dimensions.")
                return

            num_images = images.shape[0]
            print(f"Successfully loaded {num_images} images with shape {images.shape[1:]}.")

            # 2. Select a random index
            random_index = random.randint(0, num_images - 1)
            random_image = images[random_index]
            
            # 3. Determine the image display type
            # Check if the image is grayscale (2D array, e.g., (128, 128)) or color (3D array, e.g., (128, 128, 3))
            cmap_setting: Optional[str] = None
            if random_image.ndim == 2 or (random_image.ndim == 3 and random_image.shape[-1] == 1):
                cmap_setting = 'gray'
                title = f"Random Grayscale Image (Index: {random_index})"
            else:
                title = f"Random Color Image (Index: {random_index})"

            # 4. View the image
            plt.figure(figsize=(5, 5))
            # Use imshow to display the array as an image
            plt.imshow(random_image, cmap=cmap_setting)
            plt.title(title)
            plt.axis('off') # Hide axis ticks and labels for cleaner image display
            plt.show()

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

def load_and_view_largest_image(npz_file_path: str, array_key: str = 'images') -> None:
    """
    Loads an NPZ file, finds the image with the largest sum of nonzero pixel
    intensities, and displays it.
    
    Args:
        npz_file_path: The path to the .npz file.
        array_key: The key used when saving the array (default is 'images').
    """
    if not os.path.exists(npz_file_path):
        print(f"Error: The file '{npz_file_path}' was not found.")
        return

    try:
        # 1. Load the NPZ file
        print(f"Loading data from {npz_file_path} for largest image analysis...")
        
        with np.load(npz_file_path, allow_pickle=True) as data:
            if array_key not in data:
                print(f"Error: Array key '{array_key}' not found in the NPZ file.")
                return

            images = data[array_key]
            
            # --- CRITICAL ERROR CHECK ---
            # An array of images should have at least 2 dimensions: (N, H*W...)
            if images.ndim < 2:
                print(f"Error: The array '{array_key}' has an unexpected shape: {images.shape}.")
                print("Expected at least 2 dimensions (number of images, image height, ...).")
                # If the data is 1D (N,), it's not images and the function should stop.
                return
            # ---------------------------

            num_images = images.shape[0]
            print(f"Successfully loaded {num_images} images.")
            
            # 2. Calculate the sum of nonzero pixel intensities for each image
            nonzero_sums = []
            for i in range(num_images):
                img = images[i]            # object -> numpy array
                img = np.array(img)
                nonzero_sums.append(np.sum(img[img > 0]))
            nonzero_sums = np.array(nonzero_sums)

            # 3. Find largest image
            if nonzero_sums.size == 0 or np.all(nonzero_sums == 0):
                print("All images are black or have no nonzero pixels.")
                return

            max_intensity_index = np.argmax(nonzero_sums)
            largest_image = images[max_intensity_index]
            max_intensity = nonzero_sums[max_intensity_index]


            # 4. Determine the image display type (reusing logic from the first function)
            cmap_setting: Optional[str] = None
            # Check if the largest image is 2D (grayscale) or 3D with 1 channel, otherwise assume color.
            if largest_image.ndim == 2 or (largest_image.ndim == 3 and largest_image.shape[-1] == 1):
                cmap_setting = 'gray'
                title = f"Largest Grayscale Image (Index: {max_intensity_index})"
            else:
                title = f"Largest Color Image (Index: {max_intensity_index})"

            # 5. View the image
            plt.figure(figsize=(5, 5))
            plt.imshow(largest_image, cmap=cmap_setting)
            plt.title(title)
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")


if __name__ == "__main__":
    # --- Set the parameter here ---
    # FILE_TO_LOAD = 'fashionmnist.npz'
    FILE_TO_LOAD = 'imagenet.npz'
    
    # load_and_view_random_image(FILE_TO_LOAD, array_key='images')
    
    load_and_view_largest_image(FILE_TO_LOAD, array_key='images')