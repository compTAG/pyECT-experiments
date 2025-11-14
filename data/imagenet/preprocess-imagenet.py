import numpy as np
import os
from PIL import Image
from typing import List, Tuple

# --- Configuration ---
# The directory containing your JPEG images
IMAGE_DIR = 'images'
# The name of the output NPZ file
OUTPUT_FILE = '../imagenet.npz' # Changed output file name to reflect padding

# Padding value: 0.0 (black) is standard for normalized images
# Since we normalize pixels to 0.0-1.0, 0.0 is the appropriate pad value.
PADDING_VALUE = 0.0


def get_max_dimensions(directory: str) -> Tuple[int, int]:
    """
    Finds the maximum height and width among all JPEG images in the directory.
    """
    max_h, max_w = 0, 0
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            try:
                # Open the image to check dimensions without processing
                with Image.open(file_path) as img:
                    width, height = img.size
                    max_h = max(max_h, height)
                    max_w = max(max_w, width)
            except Exception as e:
                print(f"Skipping dimension check for {filename} due to error: {e}")
                
    return max_h, max_w


def load_process_and_pad_images(directory: str, max_h: int, max_w: int) -> np.ndarray:
    """
    Loads JPEG images, converts them to grayscale, normalizes them,
    pads them to (max_h, max_w), and stacks them into a single NumPy array.
    """
    print(f"Target dimensions for padding: Height={max_h}, Width={max_w}")
    # Initialize a list to hold the padded image arrays
    image_list: List[np.ndarray] = []
    processed_count = 0

    # 1. Iterate through the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)

            try:
                # 2. Open the image, convert to Grayscale, and normalize
                img = Image.open(file_path).convert('L')
                
                # Convert the Pillow image to a NumPy array (shape H, W)
                img_array = np.array(img, dtype=np.uint8)
                
                # Normalize pixel values (0-255) to (0.0-1.0)
                img_array = img_array.astype(np.float32) / 255.0
                
                current_h, current_w = img_array.shape
                
                # 3. Calculate padding
                # Pad only on the right and bottom to maintain top-left alignment
                pad_h = max_h - current_h
                pad_w = max_w - current_w
                
                # Define padding structure: ((top, bottom), (left, right))
                padding_config = ((0, pad_h), (0, pad_w))
                
                # Perform the padding
                padded_array = np.pad(
                    img_array, 
                    padding_config, 
                    mode='constant', 
                    constant_values=PADDING_VALUE
                )

                # The padded array is now guaranteed to have shape (max_h, max_w)
                image_list.append(padded_array)
                processed_count += 1

            except Exception as e:
                print(f"Skipping file {filename} due to processing/padding error: {e}")

    if not image_list:
        print(f"Error: No valid JPEG images found in the directory '{directory}'.")
        return np.array([], dtype=np.float32)

    # 4. Stack the list of uniform-sized arrays into a single NumPy array
    # The final shape will be (N, max_H, max_W)
    stacked_images = np.stack(image_list, axis=0)
    
    print(f"Successfully processed and padded {processed_count} images.")
    return stacked_images


def save_images_to_npz(images: np.ndarray, file_name: str) -> None:
    """
    Saves the stack of images to a compressed .npz file with the 'images' key.
    """
    if images.size == 0:
        print("No images to save. Aborting.")
        return

    print(f"\nSaving images to {file_name}...")
    # Save the array under the required key 'images'
    # The array now has a uniform shape (N, H, W)
    np.savez_compressed(file_name, images=images)
    print(f"Successfully saved to {file_name}. Final array shape: {images.shape}")
    print("The final array is a standard NumPy array, ready for direct use in models.")


if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.isdir(IMAGE_DIR):
        print(f"The directory '{IMAGE_DIR}' does not exist.")
        print("Please create the 'images' folder and place your JPEG files inside it.")
    else:
        # 1. Find the maximum dimensions
        max_height, max_width = get_max_dimensions(IMAGE_DIR)
        
        if max_height == 0 or max_width == 0:
            print("Could not find any valid JPEG images to determine maximum size.")
        else:
            # 2. Load, process, and pad all images to the max dimensions
            processed_images = load_process_and_pad_images(
                IMAGE_DIR, max_height, max_width
            )

            # 3. Save the transformed images
            save_images_to_npz(processed_images, OUTPUT_FILE)