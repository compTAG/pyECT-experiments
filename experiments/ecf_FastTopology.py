import torch
import argparse
import numpy as np
import sys
import time
from typing import Tuple
from tqdm import tqdm

sys.path.append('../FastTopology')
from FastTopology.py_calls_to_cpp import call_cpp_2D_parallel, call_cpp_2D
from FastTopology.EC_computation_funcs import compute_EC_curve_2D

def time_ecf_cpu(img: torch.Tensor, cores: int) -> Tuple[torch.Tensor, float]:
    img = (img - img.min()) / (img.max() - img.min())

    # convert img to numpy array
    img = img.cpu().numpy()

    # cast img to np.intc type
    img = img.astype(np.intc)

    start_time = time.perf_counter()

    if cores > 1:
        contr = call_cpp_2D_parallel(img, img.shape[0], img.shape[1], 255, cores)
    else:
        contr = call_cpp_2D(img, img.shape[0], img.shape[1], 255)

    ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
    elapsed_time = time.perf_counter() - start_time
    return ECC, elapsed_time

def compute_ecf(img: torch.Tensor, cores: int) -> Tuple[float, str]:
    ecf_values, elapsed_time = time_ecf_cpu(img, cores)

    return elapsed_time

def parse_args():
    parser = argparse.ArgumentParser(description="Run ecf experiments on image data.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the compressed numpy (.npz) file containing images.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output CSV file for results.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for image processing.")
    parser.add_argument("--cores", type=int, default=1,
                        help="Number of cores to use for parallel processing.")
    return parser.parse_args()


def main():
    args = parse_args()

    npz_data = np.load(args.image_path, allow_pickle=True)
    all_images = npz_data['images']  # shape: (num_images, H, W)
    num_images = all_images.shape[0]

    print(f"Found {num_images} images in {args.image_path}")

    cores = args.cores

    # Process images in batches
    for start_idx in tqdm(range(0, num_images, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, num_images)
        batch_images = all_images[start_idx:end_idx]

        for i, img_np in enumerate(batch_images):
            img_tensor = torch.tensor(img_np, dtype=torch.float32).to("cpu")

            elapsed_time = compute_ecf(img_tensor, cores)

            with open(args.output_path, "a") as f_out:
                row = [
                    str(start_idx + i),
                    args.image_path,
                    str(args.batch_size),
                    str(cores),
                    f"{elapsed_time:.6f}",
                ]
                f_out.write(",".join(row) + "\n")

    print(f"Finished processing. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
