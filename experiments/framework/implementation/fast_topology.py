import os
import sys
import time

import numpy as np
import torch

from domain.implementation import ImplementationResult

from usecase.interfaces import I_Implementation

# print pwd
print("Current working directory:", os.getcwd())

sys.path.append('./experiments/framework/implementation')
from FastTopology.py_calls_to_cpp import call_cpp_2D_parallel, call_cpp_2D, call_cpp_3D_parallel, call_cpp_3D
from FastTopology.EC_computation_funcs import compute_EC_curve_2D, compute_EC_curve_3D

class FastTopology_Image_ECF_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int) -> ImplementationResult:
        result = ImplementationResult()
        result.device_used = "cpu"

        # ensure data is on CPU
        data = data.cpu().numpy()

        # convert intensities from [0,1] to [0,255]
        data = (data * 255).astype(np.float32)

        # cast img to np.intc type
        data = data.astype(np.intc)

        # determine if 2D or 3D image
        if data.ndim == 2:
            t0 = time.perf_counter()
            if cores > 1:
                contr = call_cpp_2D_parallel(data, data.shape[0], data.shape[1], 255, cores)
            else:
                contr = call_cpp_2D(data, data.shape[0], data.shape[1], 255)
            ECC = compute_EC_curve_2D(contr, conn_type='8C').cumsum()
            result.computation_time = time.perf_counter() - t0
        elif data.ndim == 3:
            t0 = time.perf_counter()
            if cores > 1:
                contr = call_cpp_3D_parallel(data, data.shape[0], data.shape[1], data.shape[2], 255, cores)
            else:
                contr = call_cpp_3D(data, data.shape[0], data.shape[1], data.shape[2], 255)
            ECC = compute_EC_curve_3D(contr, conn_type='26C').cumsum()
            result.computation_time = time.perf_counter() - t0
        else:
            raise NotImplementedError("Fast Topology ECF implementation only supports 2D and 3D images.")

        # No complex construction or vectorization for fast topology ECF
        result.complex_construction_time = 0.0
        result.vectorization_time = 0.0

        result.value = ECC
        return result
