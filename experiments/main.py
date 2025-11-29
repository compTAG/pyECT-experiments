import argparse
import os
import torch

from domain.implementation import IMPLEMENTATION_NAMES
from domain.device import DEVICE_NAMES

from usecase.experiment import ExperimentRunner
from usecase.interfaces import I_Implementation

from framework.implementation.pyect import (
    PyECT_Uncompiled_WECT_CPU_Implementation,
    PyECT_Uncompiled_WECT_CUDA_Implementation,
    PyECT_Uncompiled_WECT_MPS_Implementation,
    PyECT_Uncompiled_Image_ECF_CPU_Implementation,
    PyECT_Uncompiled_Image_ECF_CUDA_Implementation,
    PyECT_Uncompiled_Image_ECF_MPS_Implementation,
    PyECT_Compiled_WECT_CPU_Implementation,
    PyECT_Compiled_WECT_CUDA_Implementation,
    PyECT_Compiled_WECT_MPS_Implementation,
    PyECT_Compiled_Image_ECF_CPU_Implementation,
    PyECT_Compiled_Image_ECF_CUDA_Implementation,
    PyECT_Compiled_Image_ECF_MPS_Implementation,
    direction_sampler_2d,
    direction_sampler_3d
)
from framework.implementation.eucalc_cpu import Eucalc_WECT_CPU_Implementation
from framework.implementation.dect import (
    DECT_WECT_CPU_Implementation,
    DECT_WECT_CUDA_Implementation,
    DECT_WECT_MPS_Implementation
)
from framework.implementation.fast_topology import FastTopology_Image_ECF_CPU_Implementation

INVARIANT_TYPES = [
    "wect",
    "ecf",
]

DATA_TYPES = [
    "image",
    "3d_mesh",
    "3d_cubical_complex"
]

IMPLEMENTATION_METHODS = {
    "wect_pyECT_uncompiled_cuda": PyECT_Uncompiled_WECT_CUDA_Implementation,
    "wect_pyECT_uncompiled_mps": PyECT_Uncompiled_WECT_MPS_Implementation,
    "wect_pyECT_uncompiled_cpu": PyECT_Uncompiled_WECT_CPU_Implementation,
    "wect_pyECT_compiled_cpu": PyECT_Compiled_WECT_CPU_Implementation,
    "wect_pyECT_compiled_cuda": PyECT_Compiled_WECT_CUDA_Implementation,
    "wect_pyECT_compiled_mps": PyECT_Compiled_WECT_MPS_Implementation,
    "wect_eucalc_cpu": Eucalc_WECT_CPU_Implementation,
    "wect_dect_cpu": DECT_WECT_CPU_Implementation,
    "wect_dect_cuda": DECT_WECT_CUDA_Implementation,
    "wect_dect_mps": DECT_WECT_MPS_Implementation,
    "ecf_pyECT_uncompiled_cuda": PyECT_Uncompiled_Image_ECF_CUDA_Implementation,
    "ecf_pyECT_uncompiled_mps": PyECT_Uncompiled_Image_ECF_MPS_Implementation,
    "ecf_pyECT_uncompiled_cpu": PyECT_Uncompiled_Image_ECF_CPU_Implementation,
    "ecf_pyECT_compiled_cpu": PyECT_Compiled_Image_ECF_CPU_Implementation,
    "ecf_pyECT_compiled_cuda": PyECT_Compiled_Image_ECF_CUDA_Implementation,
    "ecf_pyECT_compiled_mps": PyECT_Compiled_Image_ECF_MPS_Implementation,
    "ecf_fast_topology_cpu": FastTopology_Image_ECF_CPU_Implementation,
}

def get_implementation(invariant: str, implementation_name: str, device: str) -> I_Implementation:
    key = f"{invariant}_{implementation_name}_{device}"
    implementation_class = IMPLEMENTATION_METHODS.get(key)
    if implementation_class is None:
        raise NotImplementedError(f"Implementation for {key} not found.")
    return implementation_class()

def get_sampler(invariant: str, data_type: str) -> callable:
    if invariant == "wect":
        if data_type == "image":
            return direction_sampler_2d
        else:
            return direction_sampler_3d
    elif invariant == "ecf":
        def dummy_sampler(num_directions: int) -> None:
            raise ValueError("Something went wrong...trying to sample directions for the ECF case, but the image ECF does not use directions.")
        return dummy_sampler
    else:
        raise ValueError(f"Unknown invariant type: {invariant}")

def validate_args(args) -> None:
    if args.implementation_name not in IMPLEMENTATION_NAMES:
        raise ValueError(f"Invalid implementation name: {args.implementation_name}. Must be one of {IMPLEMENTATION_NAMES}.")
    
    if args.device not in DEVICE_NAMES:
        raise ValueError(f"Invalid device name: {args.device}. Must be one of {DEVICE_NAMES}.")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device specified but not available.")
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device specified but not available.")
    
    # check output path to make sure that the directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # check output path to make sure file does not already exist
    if os.path.exists(args.output_path):
        raise ValueError(f"Output file already exists. Running experiment will overwrite results: {args.output_path}")
    
    # ensure we are using a valid invariant type
    if args.invariant not in INVARIANT_TYPES:
        raise ValueError(f"Invalid invariant type: {args.invariant}. Must be one of {INVARIANT_TYPES}.")
    
    # ensure we are using a valid data type
    if args.data_type not in DATA_TYPES:
        raise ValueError(f"Invalid data type: {args.data_type}. Must be one of {DATA_TYPES}.")
    
    # ensure there are enough system cores available for what is requested
    if args.cores < 1 or args.cores > os.cpu_count():
        raise ValueError(f"Invalid number of CPU cores specified: {args.cores}. Must be between 1 and {os.cpu_count()}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run WECT experiments on image data.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the compressed numpy (.npz) or OBJ (.obj) file containing images or 3D meshes.")
    parser.add_argument("--data_type", choices=DATA_TYPES, required=True,
                        help="Type of data (e.g., 'image', '3d_mesh', '3d_cubical_complex').")
    parser.add_argument("--invariant", choices=INVARIANT_TYPES, required=True,
                        help="Type of invariant to compute (e.g., 'wect', 'ecf').")
    parser.add_argument("--implementation_name", type=str, required=True,
                        help="Name of the experiment to run.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output CSV file for results.")
    parser.add_argument("--num_directions", type=int, required=True,
                        help="Number of directions to use.")
    parser.add_argument("--num_timesteps", type=int, required=True,
                        help="Number of timesteps to use.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu",
                    help="Device to use (supports CPU, CUDA, or Apple MPS).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for image processing.")
    parser.add_argument("--cores", type=int, default=1,
                        help="Number of CPU cores to use for parallel implementations.")
    return parser.parse_args()

def main():
    args = parse_args()
    validate_args(args)

    # determine implementation
    implementation = get_implementation(args.invariant, args.implementation_name, args.device)

    # determine direction sampler
    directions_sampler = get_sampler(args.invariant, args.data_type)
    print("using sampler:", directions_sampler)

    # set up experiment
    experiment = ExperimentRunner(
        params=args,
        implementation=implementation,
        sample_directions=directions_sampler
    )

    experiment.run()

    print("Experiment completed successfully.")
    print(f"Args used: {args}")
    print("Results written to:", args.output_path)

if __name__ == "__main__":
    main()
