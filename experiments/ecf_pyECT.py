import argparse
import time

from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm

from pyect import (
    Image_ECF_2D,
)

def time_ecf_cpu(img: torch.Tensor, num_heights: int) -> Tuple[torch.Tensor, float]:
    img = (img - img.min()) / (img.max() - img.min())
    ecf = Image_ECF_2D(num_heights)
    start_time = time.perf_counter()
    ecf_values = ecf(img)
    elapsed_time = time.perf_counter() - start_time
    return ecf_values, elapsed_time


def time_ecf_cuda(img: torch.Tensor, num_heights: int) -> Tuple[torch.Tensor, float]:
    img = (img - img.min()) / (img.max() - img.min())
    ecf = Image_ECF_2D(num_heights).cuda()

    # GPU warm-up
    for _ in range(10):  # run a few times to avoid first-run overhead
        _ = ecf(img)
    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    ecf_values = ecf(img)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event) / 1000.0
    return ecf_values, elapsed_time

def time_ecf_mps(img: torch.Tensor, num_heights: int) -> Tuple[torch.Tensor, float]:
    """
    Version for Apple Silicon (Metal Performance Shaders).
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = img.to("mps")
    ecf = Image_ECF_2D(num_heights)

    # GPU warm-up
    for _ in range(10):  # run a few times to avoid first-run overhead
        _ = ecf(img)


    torch.mps.synchronize()
    start_time = time.perf_counter()
    ecf_values = ecf(img)
    torch.mps.synchronize()
    elapsed_time = time.perf_counter() - start_time
    return ecf_values, elapsed_time


def compute_ecf(img: torch.Tensor, num_heights: int) -> Tuple[float, str]:
    device_type = img.device.type

    if device_type == "cuda":
        _, elapsed_time = time_ecf_cuda(img, num_heights)
    elif device_type == "mps":
        _, elapsed_time = time_ecf_mps(img,  num_heights)
    else:
        _, elapsed_time = time_ecf_cpu(img, num_heights)

    return elapsed_time, device_type

def parse_args():
    parser = argparse.ArgumentParser(description="Run ecf experiments on image data.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the compressed numpy (.npz) file containing images.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output CSV file for results.")
    parser.add_argument("--num_timesteps", type=int, required=True,
                        help="Number of timesteps to use.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu",
                    help="Device to use (supports CPU, CUDA, or Apple MPS).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for image processing.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    npz_data = np.load(args.image_path, allow_pickle=True)
    all_images = npz_data['images']  # shape: (num_images, H, W)
    num_images = all_images.shape[0]

    print(f"Found {num_images} images in {args.image_path}")

    # Process images in batches
    for start_idx in tqdm(range(0, num_images, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, num_images)
        batch_images = all_images[start_idx:end_idx]

        for i, img_np in enumerate(batch_images):
            img_tensor = torch.tensor(img_np, dtype=torch.float32).to(device)

            elapsed_time, device_type = compute_ecf(img_tensor, args.num_timesteps)

            with open(args.output_path, "a") as f_out:
                row = [
                    str(start_idx + i),
                    args.image_path,
                    str(args.num_timesteps),
                    args.device,
                    str(args.batch_size),
                    f"{elapsed_time:.6f}",
                    device_type,
                ]
                f_out.write(",".join(row) + "\n")

    print(f"Finished processing. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
