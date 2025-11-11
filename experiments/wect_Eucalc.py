import argparse
import time

from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm

from pyect import (
    sample_directions_2d
)

import sys
sys.path.append('./experiments/eucalc')
import eucalc

def time_wect_cpu(img: np.array, dirs: torch.Tensor, num_heights: int) -> Tuple[torch.Tensor, float]:
    img = (img - img.min()) / (img.max() - img.min())
    cplx = eucalc.EmbeddedComplex(img)
    start_time = time.perf_counter()
    cplx.preproc_ect()

    for direction in dirs:
        ect_dir = cplx.compute_euler_characteristic_transform(direction)
    elapsed_time = time.perf_counter() - start_time

    return None, elapsed_time

def compute_wect(img: np.array, dirs: torch.Tensor, num_heights: int) -> Tuple[float, str]:
    _, elapsed_time = time_wect_cpu(img, dirs, num_heights)

    return elapsed_time, "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Run WECT experiments on image data.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the compressed numpy (.npz) file containing images.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output CSV file for results.")
    parser.add_argument("--num_directions", type=int, required=True,
                        help="Number of directions to use.")
    parser.add_argument("--num_timesteps", type=int, required=True,
                        help="Number of timesteps to use.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for image processing.")
    return parser.parse_args()


def main():
    args = parse_args()

    dirs = sample_directions_2d(args.num_directions).to("cpu").numpy()


    npz_data = np.load(args.image_path, allow_pickle=True)
    all_images = npz_data['images']  # shape: (num_images, H, W)
    num_images = all_images.shape[0]

    print(f"Found {num_images} images in {args.image_path}")

    # Process images in batches
    for start_idx in tqdm(range(0, num_images, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, num_images)
        batch_images = all_images[start_idx:end_idx]

        for i, img_np in enumerate(batch_images):

            elapsed_time, device_type = compute_wect(img_np, dirs, args.num_timesteps)

            with open(args.output_path, "a") as f_out:
                row = [
                    str(start_idx + i),
                    args.image_path,
                    str(args.num_directions),
                    str(args.num_timesteps),
                    str(args.batch_size),
                    f"{elapsed_time:.6f}",
                    device_type,
                ]
                f_out.write(",".join(row) + "\n")

    print(f"Finished processing. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
