import numpy as np
import torch
from tqdm import tqdm

from domain.implementation import ImplementationResult
from domain.experiment import ExperimentParameters
from domain.device import DEVICE_NAMES
from .interfaces import I_Implementation

class ExperimentRunner:
    def __init__(self, params: ExperimentParameters, implementation: I_Implementation, sample_directions: callable):
        # expand parameters
        self.device = params.device
        self.data_path = params.data_path
        self.batch_size = params.batch_size
        self.invariant = params.invariant
        self.num_directions = params.num_directions
        self.num_timesteps = params.num_timesteps
        self.output_path = params.output_path
        self.cores = params.cores

        # store implementation and direction sampler
        self.implementation = implementation
        self.sample_directions = sample_directions

    def run(self) -> ImplementationResult:

        if self.invariant == "wect":
            dirs = self.sample_directions(self.num_directions).to(self.device)
        else:
            dirs = None  # Other variants do not use directions

        # load data
        npz_data = np.load(self.data_path, allow_pickle=True)
        all_images = npz_data['images']  # shape: (num_images, H, W)
        num_images = all_images.shape[0]
        print(f"Found {num_images} images in {self.data_path}")

        # write header to output file
        with open(self.output_path, "w") as f_out:
            header = [
                "image_index",
                "data_path",
                "num_directions",
                "num_timesteps",
                "device_requested",
                "device_used",
                "batch_size",
                "num_cores",
                "complex_construction_time",
                "computation_time",
                "vectorization_time",
            ]
            f_out.write(",".join(header) + "\n")

        # iterate over data, compute the results and write to output file
        for start_idx in tqdm(range(0, num_images, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_images)
            batch_images = all_images[start_idx:end_idx]

            for i, img_np in enumerate(batch_images):
                img_tensor = torch.tensor(img_np, dtype=torch.float32).to(self.device)

                result = self.implementation.compute(img_tensor, dirs, self.num_timesteps, self.cores)

                with open(self.output_path, "a") as f_out:
                    row = [
                        str(start_idx + i),
                        self.data_path,
                        str(self.num_directions),
                        str(self.num_timesteps),
                        self.device,
                        result.device_used,
                        str(self.batch_size),
                        str(self.cores),  # num_cores is fixed to 1 for now
                        f"{result.complex_construction_time:.6f}",
                        f"{result.computation_time:.6f}",
                        f"{result.vectorization_time:.6f}",
                    ]
                    f_out.write(",".join(row) + "\n")
