import time
import torch

from usecase.interfaces import I_Implementation
from domain.implementation import ImplementationResult

from pyect import weighted_cubical
from dect.ect import compute_ect
from dect.ect_fn import indicator

class DECT_WECT_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int,
                cores: int = 1, data_type: str = "image") -> ImplementationResult:

        result = ImplementationResult()
        result.device_used = "cpu"
        torch.set_grad_enabled(False)

        # 1. Convert image → DECT complex (x, simplices)
        t0 = time.perf_counter()

        if data_type != "image":
            raise NotImplementedError("DECT currently supports data_type='image' only")

        img_complex = weighted_cubical(data)   # CPU version
        x = img_complex.dimensions[0][0]
        edges = img_complex.dimensions[1][0]
        triangles = img_complex.dimensions[2][0]
        simplices = (edges, triangles)

        # radius based on image size
        H, W = data.shape[-2], data.shape[-1]
        radius = ((0.5 * H) ** 2 + (0.5 * W) ** 2) ** 0.5

        result.complex_construction_time = time.perf_counter() - t0

        # directions need to be transposed for DECT
        v = directions.T.contiguous()

        # 2. Compute DECT
        t1 = time.perf_counter()
        values = compute_ect(
            x,
            v=v,
            radius=radius,
            resolution=num_heights,
            scale=1,
            ect_fn=indicator
        )
        result.computation_time = time.perf_counter() - t1

        result.vectorization_time = 0.0
        result.value = values.cpu()
        return result

class DECT_WECT_CUDA_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int,
                cores: int = 1, data_type: str = "image") -> ImplementationResult:

        result = ImplementationResult()
        result.device_used = "cuda"
        torch.set_grad_enabled(False)

        data = data.cuda()
        directions = directions.cuda()

        # 1. Complex construction (GPU)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if data_type != "image":
            raise NotImplementedError("DECT currently supports data_type='image' only")

        img_complex = weighted_cubical(data)
        x = img_complex.dimensions[0][0]
        edges = img_complex.dimensions[1][0]
        triangles = img_complex.dimensions[2][0]
        simplices = (edges, triangles)

        H, W = data.shape[-2], data.shape[-1]
        radius = ((0.5 * H) ** 2 + (0.5 * W) ** 2) ** 0.5

        end_event.record()
        torch.cuda.synchronize()
        result.complex_construction_time = start_event.elapsed_time(end_event) / 1000.0

        v = directions.T.contiguous()

        # 2. Compute DECT
        start_event.record()
        values = compute_ect(
            x,
            v=v,
            radius=radius,
            resolution=num_heights,
            scale=1,
            ect_fn=indicator
        )
        end_event.record()
        torch.cuda.synchronize()

        result.computation_time = start_event.elapsed_time(end_event) / 1000.0
        result.vectorization_time = 0.0
        result.value = values  # keep on GPU
        return result

class DECT_WECT_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int,
                cores: int = 1, data_type: str = "image") -> ImplementationResult:

        result = ImplementationResult()
        result.device_used = "mps"
        torch.set_grad_enabled(False)

        data = data.to("mps")
        directions = directions.to("mps")

        # 1. Complex construction on MPS
        torch.mps.synchronize()
        t0 = time.perf_counter()

        if data_type != "image":
            raise NotImplementedError("DECT currently supports data_type='image' only")

        img_complex = weighted_cubical(data)
        x = img_complex.dimensions[0][0]
        edges = img_complex.dimensions[1][0]
        triangles = img_complex.dimensions[2][0]
        simplices = (edges, triangles)

        H, W = data.shape[-2], data.shape[-1]
        radius = ((0.5 * H) ** 2 + (0.5 * W) ** 2) ** 0.5

        torch.mps.synchronize()
        result.complex_construction_time = time.perf_counter() - t0

        v = directions.T.contiguous()

        # 2. Compute DECT
        torch.mps.synchronize()
        t1 = time.perf_counter()

        values = compute_ect(
            x,
            v=v,
            radius=radius,
            resolution=num_heights,
            scale=1,
            ect_fn=indicator
        )

        torch.mps.synchronize()
        result.computation_time = time.perf_counter() - t1

        result.vectorization_time = 0.0
        result.value = values.cpu()
        return result

