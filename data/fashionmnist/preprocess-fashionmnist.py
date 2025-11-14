import pandas as pd
import numpy as np

# Fashion-MNIST images are 28x28 pixels
IMAGE_SIZE = 28
# The dataset has 784 pixel columns (28 * 28)
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# The label column is the first one
LABEL_COL = 'label'


def load_and_transform_images(csv_file_path: str) -> np.ndarray:
    """
    Loads Fashion-MNIST data, extracts and normalizes pixel data, 
    and reshapes it into a stack of 28x28 images.
    """
    print(f"Loading data from {csv_file_path}...")
    # 1. Load the CSV file
    df = pd.read_csv(csv_file_path)

    # 2. Forget the 'label' column and select only pixel data
    # Pixel data columns start from 'pixel1' up to 'pixel784'
    # We use .iloc to select all rows and all columns *except* the first one (index 0)
    pixel_data = df.iloc[:, 1:].values 
    
    # 3. Reshape the data
    # The shape should be (number_of_images, 28, 28)
    num_images = pixel_data.shape[0]
    
    # We convert the flat array of 784 pixels per image into a 28x28 matrix
    images = pixel_data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
    
    # Optional: Normalize the pixel values from 0-255 to 0.0-1.0
    images = images.astype(np.float32) / 255.0
    
    print(f"Data transformed. Total images: {num_images}, Image shape: ({IMAGE_SIZE}, {IMAGE_SIZE})")
    
    return images


def save_images_to_npz(images: np.ndarray, file_name: str) -> None:
    """
    Saves the stack of images to a compressed .npz file.
    """
    print(f"Saving images to {file_name}...")
    # Save the array under the key 'images'
    np.savez_compressed(file_name, images=images)
    print(f"Saved images to {file_name} (final array shape: {images.shape})")


if __name__ == "__main__":
    # Define the input and output file paths
    INPUT_CSV = 'fashion-mnist_train.csv'
    OUTPUT_NPZ = '../fashionmnist.npz'
    
    # Load and transform the data
    transformed_images = load_and_transform_images(INPUT_CSV)
    
    # Save the transformed images
    save_images_to_npz(transformed_images, OUTPUT_NPZ)
    
    # --- Verification (Optional) ---
    print("\n--- Verification ---")
    
    # Load the saved file to check the structure
    with np.load(OUTPUT_NPZ) as data:
        loaded_images = data['images']
        print(f"Successfully loaded {OUTPUT_NPZ}")
        print(f"Loaded array shape: {loaded_images.shape}")
        
        # Displaying the first image's shape (should be 28, 28)
        if loaded_images.shape[0] > 0:
            print(f"Shape of the first image: {loaded_images[0].shape}")
        
        # Print a small section of the first image data for a quick check
        print("Snippet of the first image's data (top-left 5x5):")
        print(loaded_images[0][:5, :5])