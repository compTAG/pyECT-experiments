"""
experiments/gvox_s.py

Measures ECF kernel throughput in GVox/s (Giga Voxels per second) for
Image_ECF_2D and Image_ECF_3D with all data pre-loaded to GPU memory.

GVox/s is defined as:
    (number of input voxels) / (wall-clock seconds for one ECF forward pass) / 1e9

where "input voxels" = H*W for 2D inputs, D*H*W for 3D inputs.

Usage:
    python experiments/gvox_s.py [--device cuda|mps|cpu] [--output PATH]
"""

import argparse
import csv
import time

import numpy as np
import torch
from pyect import Image_ECF_2D, Image_ECF_3D


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure_ecf_time(ecf, data: torch.Tensor, device: str,
                     n_warmup: int = 20, n_repeats: int = 50) -> float:
    """Return median wall-clock time (seconds) for one ECF forward pass.

    Data must already reside on *device* before calling this function.
    CUDA timing uses hardware events; MPS/CPU use perf_counter with sync.
    """
    # warm-up: let JIT / driver reach steady state
    for _ in range(n_warmup):
        _ = ecf(data)
    sync(device)

    if device == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(n_repeats):
            start_ev.record()
            _ = ecf(data)
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev) / 1000.0)  # ms -> s
    else:
        times = []
        for _ in range(n_repeats):
            sync(device)
            t0 = time.perf_counter()
            _ = ecf(data)
            sync(device)
            times.append(time.perf_counter() - t0)

    return float(np.median(times))

# 2D configs: (H, W, num_thresholds)
CONFIGS_2D = [
    (2048, 2048,  100),
    (2048, 2048, 1000),
    (2048, 2048, 10000),
    (4096, 4096,  100),
    (4096, 4096, 1000),
    (4096, 4096, 10000),
    (8192, 8192,  100),
    (8192, 8192, 1000),
    (8192, 8192, 10000),
]

# 3D configs: (D, H, W, num_thresholds)
CONFIGS_3D = [
    (128, 128, 128,  100),
    (128, 128, 128, 1000),
    (128, 128, 128, 10000),
    (256, 256, 256,  100),
    (256, 256, 256, 1000),
    (256, 256, 256, 10000),
    (512, 512, 512,  100),
    (512, 512, 512, 1000),
    (512, 512, 512, 10000),
]


def make_ecf(ecf_cls, T: int, compiled: bool):
    ecf = ecf_cls(T).eval()
    if compiled:
        ecf = torch.compile(
            ecf,
            backend="inductor",
            mode="max-autotune",
            dynamic=False,
        )
    return ecf


def run_2d(device: str, compiled: bool) -> list:
    results = []
    label = "compiled" if compiled else "uncompiled"
    header = f"{'Size':>10}  {'Thresholds':>10}  {'Median (ms)':>12}  {'GVox/s':>10}"
    print(f"\n{'='*len(header)}")
    print(f"Image_ECF_2D  ({label})  —  device: {device}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for H, W, T in CONFIGS_2D:
        ecf = make_ecf(Image_ECF_2D, T, compiled)
        # pre-allocate on device so no transfer occurs in the timed region
        data = torch.rand(H, W, dtype=torch.float32, device=device)

        t_sec = measure_ecf_time(ecf, data, device)
        gvox_s = (H * W) / t_sec / 1e9

        size_str = f"{H}x{W}"
        print(f"{size_str:>10}  {T:>10}  {t_sec * 1000:>12.3f}  {gvox_s:>10.4f}")
        results.append(
            {"dim": "2D", "compiled": label, "size": size_str, "thresholds": T,
             "time_ms": round(t_sec * 1000, 4), "gvox_s": round(gvox_s, 6)}
        )

    return results


def run_3d(device: str, compiled: bool) -> list:
    results = []
    label = "compiled" if compiled else "uncompiled"
    header = f"{'Size':>14}  {'Thresholds':>10}  {'Median (ms)':>12}  {'GVox/s':>10}"
    print(f"\n{'='*len(header)}")
    print(f"Image_ECF_3D  ({label})  —  device: {device}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for D, H, W, T in CONFIGS_3D:
        ecf = make_ecf(Image_ECF_3D, T, compiled)
        data = torch.rand(D, H, W, dtype=torch.float32, device=device)

        t_sec = measure_ecf_time(ecf, data, device)
        gvox_s = (D * H * W) / t_sec / 1e9

        size_str = f"{D}x{H}x{W}"
        print(f"{size_str:>14}  {T:>10}  {t_sec * 1000:>12.3f}  {gvox_s:>10.4f}")
        results.append(
            {"dim": "3D", "compiled": label, "size": size_str, "thresholds": T,
             "time_ms": round(t_sec * 1000, 4), "gvox_s": round(gvox_s, 6)}
        )

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ECF kernel throughput in GVox/s (data pre-loaded to GPU)."
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default=None,
        help="Compute device (default: auto-detect)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write a results CSV file."
    )
    parser.add_argument(
        "--skip_3d", action="store_true",
        help="Skip 3D benchmarks (useful for quick checks or memory-limited devices)."
    )
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    torch.set_grad_enabled(False)

    results = run_2d(device, compiled=False)
    results += run_2d(device, compiled=True)
    if not args.skip_3d:
        results += run_3d(device, compiled=False)
        results += run_3d(device, compiled=True)

    # summary
    gvox_vals = [r["gvox_s"] for r in results]
    sep = "=" * 48
    print(f"\n{sep}")
    print(f"  GVox/s summary across all configurations")
    print(f"    Min:    {min(gvox_vals):.4f}")
    print(f"    Max:    {max(gvox_vals):.4f}")
    print(f"    Median: {float(np.median(gvox_vals)):.4f}")
    print(sep)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["dim", "compiled", "size", "thresholds", "time_ms", "gvox_s"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
