import sys
import time

import torch

from domain.implementation import ImplementationResult

from usecase.interfaces import I_Implementation

sys.path.append('./experiments/framework/implementation/eucalc')
import eucalc

class Eucalc_WECT_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int = 1) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cpu"

        # convert data to numpy arrays
        data = data.cpu().numpy()
        directions = directions.cpu().numpy()

        t0 = time.perf_counter()
        cplx = eucalc.EmbeddedComplex(data)
        result.complex_construction_time = time.perf_counter() - t0

        t1 = time.perf_counter()

        # Preprocessing
        cplx.preproc_ect()

        # Compute ECT per direction (no storing — original returned None)
        for direction in directions:
            _ = cplx.compute_euler_characteristic_transform(direction)

        result.computation_time = time.perf_counter() - t1


        # time an iteration over evenly spaced height values
        heights = torch.linspace(0, 1, num_heights)
        running_time = 0.0
        for direction in directions:
            ecc = cplx.compute_euler_characteristic_transform(direction)
            t2 = time.perf_counter()
            for h in heights:
                _ = ecc.evaluate(h.item())
            running_time += time.perf_counter() - t2

        result.vectorization_time = running_time

        result.value = None

        return result
