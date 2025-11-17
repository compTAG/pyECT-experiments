import time

import torch

from usecase.interfaces import I_Implementation 
from domain.implementation import ImplementationResult
from pyect import WECT, weighted_cubical, sample_directions_2d, sample_directions_3d, mesh_to_complex

def direction_sampler_2d(num_directions: int) -> torch.Tensor:
    return sample_directions_2d(num_directions)

def direction_sampler_3d(num_directions: int) -> torch.Tensor:
    return sample_directions_3d(num_directions)

######################################################################
#                 UNCOMPILED PYECT WECT IMPLEMENTATIONS
######################################################################
class PyECT_Uncompiled_WECT_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image") -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cpu"

        torch.set_grad_enabled(False)

        # 1. Construct complex
        if data_type == "image":
            t0 = time.perf_counter()
            cmplx = weighted_cubical(data)
            result.complex_construction_time = time.perf_counter() - t0
        elif data_type == "3d_mesh":
            t0 = time.perf_counter()
            cmplx = mesh_to_complex(data)
            result.complex_construction_time = time.perf_counter() - t0
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for WECT.")

        # 2. Compute WECT
        wect = WECT(directions, num_heights).eval()
        t1 = time.perf_counter()
        values = wect(cmplx)
        result.computation_time = time.perf_counter() - t1

        # pyect already produces a vectorization
        result.vectorization_time = 0.0

        result.value = values.cpu()
        return result

class PyECT_Uncompiled_WECT_CUDA_Implementation(I_Implementation):
    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image") -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cuda"

        torch.set_grad_enabled(False)

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

        # 3D mesh support added
        if data_type == "image":
            cmplx, t_complex = cuda_timing(lambda: weighted_cubical(data))
        elif data_type == "3d_mesh":
            cmplx, t_complex = cuda_timing(lambda: mesh_to_complex(data))
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for WECT.")

        result.complex_construction_time = t_complex

        wect = WECT(directions, num_heights).eval().cuda()

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


class PyECT_Uncompiled_WECT_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image") -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "mps"

        torch.set_grad_enabled(False)

        directions = directions.to("mps")
        data = data.to("mps")

        # 1. Construct complex
        torch.mps.synchronize()
        t0 = time.perf_counter()

        # 3D mesh support added
        if data_type == "image":
            cmplx = weighted_cubical(data)
        elif data_type == "3d_mesh":
            cmplx = mesh_to_complex(data)
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for WECT.")

        torch.mps.synchronize()
        result.complex_construction_time = time.perf_counter() - t0

        # 2. Construct WECT
        torch.mps.synchronize()
        wect = WECT(directions, num_heights).eval()

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


from pyect import Image_ECF_2D, Image_ECF_3D

######################################################################
#                 UNCOMPILED PYECT IMAGE-ECF IMPLEMENTATIONS
######################################################################

class PyECT_Uncompiled_Image_ECF_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image") -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cpu"

        torch.set_grad_enabled(False)

        if data_type != "image" and data_type != "3d_cubical_complex":
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # check if 2d or 3d image
        if data.dim() == 2:
            ecf = Image_ECF_2D(num_heights).eval()
        elif data.dim() == 3:
            ecf = Image_ECF_3D(num_heights).eval()
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

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


class PyECT_Uncompiled_Image_ECF_CUDA_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image") -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cuda"

        torch.set_grad_enabled(False)

        # Move data to GPU
        data = data.cuda()

        if data_type != "image" and data_type != "3d_cubical_complex":
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # check if 2d or 3d image
        if data.dim() == 2:
            ecf = Image_ECF_2D(num_heights).eval()
        elif data.dim() == 3:
            ecf = Image_ECF_3D(num_heights).eval()
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # GPU warm-up
        for _ in range(100):
            _ = ecf(data)
        torch.cuda.synchronize()

        # Timing with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        ecf_values = ecf(data)
        end_event.record()
        torch.cuda.synchronize()

        result.computation_time = start_event.elapsed_time(end_event) / 1000.0

        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0

        result.value = ecf_values  # stays on GPU, consistent with WECT-CUDA class
        return result


class PyECT_Uncompiled_Image_ECF_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image") -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "mps"

        torch.set_grad_enabled(False)

        # Move to MPS
        data = data.to("mps")

        if data_type != "image" and data_type != "3d_cubical_complex":
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # check if 2d or 3d image
        if data.dim() == 2:
            ecf = Image_ECF_2D(num_heights).eval()
        elif data.dim() == 3:
            ecf = Image_ECF_3D(num_heights).eval()
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # Warm-up
        for _ in range(100):
            _ = ecf(data)

        torch.mps.synchronize()

        # Timing
        t0 = time.perf_counter()
        ecf_values = ecf(data)
        torch.mps.synchronize()
        result.computation_time = time.perf_counter() - t0

        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0

        result.value = ecf_values.cpu()
        return result

######################################################################
#                   COMPILED PYECT WECT IMPLEMENTATIONS
######################################################################

class PyECT_Compiled_WECT_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image"):
        result = ImplementationResult()
        result.device_used = "cpu"

        torch.set_grad_enabled(False)

        # 1. Construct complex
        t0 = time.perf_counter()

        # 3D mesh support added
        if data_type == "image":
            cmplx = weighted_cubical(data)
        elif data_type == "3d_mesh":
            cmplx = mesh_to_complex(data)
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for WECT.")

        result.complex_construction_time = time.perf_counter() - t0

        # 2. Compiled WECT
        wect = WECT(directions, num_heights).eval()
        compiled_wect = torch.compile(
            wect,
            backend="inductor",
            mode="max-autotune",
            dynamic=True
        )

        # Warm-up
        for _ in range(100):
            _ = compiled_wect(cmplx)

        # Compute
        t1 = time.perf_counter()
        values = compiled_wect(cmplx)
        result.computation_time = time.perf_counter() - t1

        result.vectorization_time = 0.0
        result.value = values.cpu()
        return result


class PyECT_Compiled_WECT_CUDA_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image"):
        result = ImplementationResult()
        result.device_used = "cuda"

        torch.set_grad_enabled(False)

        data = data.cuda()
        directions = directions.cuda()

        # Complex construction
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # 3D mesh support added
        if data_type == "image":
            cmplx = weighted_cubical(data)
        elif data_type == "3d_mesh":
            cmplx = mesh_to_complex(data)
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for WECT.")

        end.record()
        torch.cuda.synchronize()
        result.complex_construction_time = start.elapsed_time(end) / 1000.0

        # Compile WECT
        wect = WECT(directions, num_heights).eval().cuda()
        compiled_wect = torch.compile(
            wect,
            backend="inductor",
            mode="max-autotune",
            dynamic=True
        )

        # Warm-up
        for _ in range(100):
            _ = compiled_wect(cmplx)
        torch.cuda.synchronize()

        start.record()
        values = compiled_wect(cmplx)
        end.record()
        torch.cuda.synchronize()

        result.computation_time = start.elapsed_time(end) / 1000.0
        result.vectorization_time = 0.0
        result.value = values  # stays on GPU
        return result


class PyECT_Compiled_WECT_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image"):
        result = ImplementationResult()
        result.device_used = "mps"

        torch.set_grad_enabled(False)

        data = data.to("mps")
        directions = directions.to("mps")

        # Construct complex
        torch.mps.synchronize()
        t0 = time.perf_counter()

        # 3D mesh support added
        if data_type == "image":
            cmplx = weighted_cubical(data)
        elif data_type == "3d_mesh":
            cmplx = mesh_to_complex(data)
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for WECT.")

        torch.mps.synchronize()
        result.complex_construction_time = time.perf_counter() - t0

        # Compile WECT
        wect = WECT(directions, num_heights).eval().to("mps")
        compiled_wect = torch.compile(
            wect,
            backend="inductor",
            mode="max-autotune",
            dynamic=True
        )

        # Warm-up
        for _ in range(100):
            _ = compiled_wect(cmplx)
        torch.mps.synchronize()

        t1 = time.perf_counter()
        values = compiled_wect(cmplx)
        torch.mps.synchronize()
        result.computation_time = time.perf_counter() - t1

        result.vectorization_time = 0.0
        result.value = values.cpu()
        return result


######################################################################
#                   COMPILED PYECT IMAGE-ECF IMPLEMENTATIONS
######################################################################

class PyECT_Compiled_Image_ECF_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image"):
        result = ImplementationResult()
        result.device_used = "cpu"

        torch.set_grad_enabled(False)

        if data_type != "image" and data_type != "3d_cubical_complex":
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # check if 2d or 3d image
        if data.dim() == 2:
            ecf = Image_ECF_2D(num_heights).eval()
        elif data.dim() == 3:
            ecf = Image_ECF_3D(num_heights).eval()
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")
        
        compiled_ecf = torch.compile(
            ecf,
            backend="inductor",
            mode="max-autotune",
            dynamic=False
        )

        # Warm-up
        for _ in range(100):
            _ = compiled_ecf(data)

        t0 = time.perf_counter()
        ecf_vals = compiled_ecf(data)
        result.computation_time = time.perf_counter() - t0

        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0
        result.value = ecf_vals.cpu()
        return result


class PyECT_Compiled_Image_ECF_CUDA_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image"):
        result = ImplementationResult()
        result.device_used = "cuda"

        torch.set_grad_enabled(False)

        data = data.cuda()

        if data_type != "image" and data_type != "3d_cubical_complex":
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # check if 2d or 3d image
        if data.dim() == 2:
            ecf = Image_ECF_2D(num_heights).eval()
        elif data.dim() == 3:
            ecf = Image_ECF_3D(num_heights).eval()
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")
        
        compiled_ecf = torch.compile(
            ecf,
            backend="inductor",
            mode="max-autotune",
            dynamic=False
        )

        for _ in range(100):
            _ = compiled_ecf(data)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        vals = compiled_ecf(data)
        end.record()
        torch.cuda.synchronize()

        result.computation_time = start.elapsed_time(end) / 1000.0
        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0
        result.value = vals  # keep on GPU
        return result


class PyECT_Compiled_Image_ECF_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1, data_type: str = "image"):
        result = ImplementationResult()
        result.device_used = "mps"

        torch.set_grad_enabled(False)

        data = data.to("mps")

        if data_type != "image" and data_type != "3d_cubical_complex":
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")

        # check if 2d or 3d image
        if data.dim() == 2:
            ecf = Image_ECF_2D(num_heights).eval()
        elif data.dim() == 3:
            ecf = Image_ECF_3D(num_heights).eval()
        else:
            raise NotImplementedError(f"Data type {data_type} not supported for Image-ECF.")
        
        compiled_ecf = torch.compile(
            ecf,
            backend="inductor",
            mode="max-autotune",
            dynamic=False
        )

        # Warm-up
        for _ in range(100):
            _ = compiled_ecf(data)
        torch.mps.synchronize()

        t0 = time.perf_counter()
        vals = compiled_ecf(data)
        torch.mps.synchronize()
        result.computation_time = time.perf_counter() - t0

        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0
        result.value = vals.cpu()
        return result
