import time

import torch

from usecase.interfaces import I_Implementation 
from domain.implementation import ImplementationResult
from pyect import WECT, weighted_freudenthal, sample_directions_2d, sample_directions_3d

def direction_sampler_2d(num_directions: int) -> torch.Tensor:
    return sample_directions_2d(num_directions)

def direction_sampler_3d(num_directions: int) -> torch.Tensor:
    return sample_directions_3d(num_directions)

class PyECT_WECT_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cpu"

        # Normalize
        data_norm = (data - data.min()) / (data.max() - data.min())

        # 1. Construct complex
        t0 = time.perf_counter()
        cmplx = weighted_freudenthal(data_norm)
        result.complex_construction_time = time.perf_counter() - t0

        # 2. Compute WECT
        wect = WECT(directions, num_heights)
        t1 = time.perf_counter()
        values = wect(cmplx)
        result.computation_time = time.perf_counter() - t1

        # pyect already produces a vectorization
        result.vectorization_time = 0.0

        result.value = values.cpu()
        return result

class PyECT_WECT_CUDA_Implementation(I_Implementation):
    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cuda"

        # ensure all tensors are on GPU
        data = data.cuda()
        directions = directions.cuda()

        # Utility to time GPU steps with CUDA events
        def cuda_timing(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize()
            return output, start.elapsed_time(end) / 1000.0  # seconds

        cmplx, t_complex = cuda_timing(lambda: weighted_freudenthal(data))
        result.complex_construction_time = t_complex

        wect = WECT(directions, num_heights).cuda()

        for _ in range(100):
            _ = wect(cmplx)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        values = wect(cmplx)
        end_event.record()
        torch.cuda.synchronize()

        result.computation_time = start_event.elapsed_time(end_event) / 1000.0  # seconds

        # pyect already produces a vectorization
        result.vectorization_time = 0.0

        result.values = values

        return result


class PyECT_WECT_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "mps"

        directions = directions.to("mps")
        data = data.to("mps")

        # 1. Construct complex
        torch.mps.synchronize()
        t0 = time.perf_counter()
        cmplx = weighted_freudenthal(data)
        torch.mps.synchronize()
        result.complex_construction_time = time.perf_counter() - t0

        # 2. Construct WECT
        torch.mps.synchronize()
        wect = WECT(directions, num_heights)

        # warm-up
        for _ in range(100):
            _ = wect(cmplx)
        torch.mps.synchronize()

        # 3. Compute
        torch.mps.synchronize()
        t1 = time.perf_counter()
        values = wect(cmplx)
        torch.mps.synchronize()
        result.computation_time = time.perf_counter() - t1

        result.vectorization_time = 0.0
        result.value = values.cpu()
        return result


from pyect import Image_ECF_2D


class PyECT_Image_ECF_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cpu"

        # Construct ECF object (negligible time, counted as computation)
        ecf = Image_ECF_2D(num_heights)

        # Timing
        t0 = time.perf_counter()
        ecf_values = ecf(data)
        result.computation_time = time.perf_counter() - t0

        # No complex construction for Image-ECF
        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0

        result.value = ecf_values.cpu()
        return result


class PyECT_Image_ECF_CUDA_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cuda"

        # Move data to GPU
        img = (data - data.min()) / (data.max() - data.min())
        img = img.cuda()

        # Construct ECF object
        ecf = Image_ECF_2D(num_heights).cuda()

        # GPU warm-up
        for _ in range(100):
            _ = ecf(img)
        torch.cuda.synchronize()

        # Timing with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        ecf_values = ecf(img)
        end_event.record()
        torch.cuda.synchronize()

        result.computation_time = start_event.elapsed_time(end_event) / 1000.0

        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0

        result.value = ecf_values  # stays on GPU, consistent with WECT-CUDA class
        return result


class PyECT_Image_ECF_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "mps"

        # Normalize + move to MPS
        img = (data - data.min()) / (data.max() - data.min())
        img = img.to("mps")

        # Construct ECF object
        ecf = Image_ECF_2D(num_heights)

        # Warm-up
        for _ in range(100):
            _ = ecf(img)

        torch.mps.synchronize()

        # Timing
        t0 = time.perf_counter()
        ecf_values = ecf(img)
        torch.mps.synchronize()
        result.computation_time = time.perf_counter() - t0

        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0

        result.value = ecf_values.cpu()
        return result
