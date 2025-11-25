import time
import torch

from usecase.interfaces import I_Implementation
from domain.implementation import ImplementationResult

from pyect import weighted_cubical, mesh_to_complex
from dect.ect import compute_ect, compute_ect_mesh
from dect.ect_fn import indicator

class DECT_WECT_CPU_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int,
                cores: int = 1, data_type: str = "image") -> ImplementationResult:

        result = ImplementationResult()
        result.device_used = "cpu"
        torch.set_grad_enabled(False)

        t0 = time.perf_counter()

        if data_type == "image":

            # existing image pipeline
            img_complex = weighted_cubical(data)   # CPU version

            x = img_complex.dimensions[0][0]           # [N, D]
            edge_index = img_complex.dimensions[1][0]  # [num_edges, 2]
            face_index = img_complex.dimensions[2][0]  # [num_faces, 3]

            simplices = (edge_index, face_index)

            # radius based on image size
            H, W = data.shape[-2], data.shape[-1]
            radius = ((0.5 * H) ** 2 + (0.5 * W) ** 2) ** 0.5

        elif data_type == "3d_mesh":
            # mesh_to_complex loads vertices / edges / triangles already on CPU
            mesh_complex = mesh_to_complex(data, device=torch.device("cpu"))

            x = mesh_complex.dimensions[0][0]          # [N, 3]
            edge_index = mesh_complex.dimensions[1][0].T  # convert to [2, num_edges]
            face_index = mesh_complex.dimensions[2][0].T  # convert to [3, num_faces]

            simplices = (edge_index, face_index)

            # radius = bounding sphere radius of point cloud
            # (this matches the intent of image radius: half diag)
            mins = x.min(dim=0).values
            maxs = x.max(dim=0).values
            center = 0.5 * (mins + maxs)
            radius = ( ((x - center) ** 2).sum(dim=1).sqrt().max().item() )

        else:
            raise NotImplementedError(
                "DECT supports data_type='image' and '3d_mesh'"
            )

        result.complex_construction_time = time.perf_counter() - t0

        # directions need to be transposed for DECT
        v = directions.T.contiguous()

        # 2. Compute DECT
        t1 = time.perf_counter()

        if data_type == "3d_mesh":
            values = compute_ect_mesh(
                x=x,
                edge_index=edge_index,
                face_index=face_index,
                v=v,
                radius=radius,
                resolution=num_heights,
                scale=1.0,
                index=None,
            )
        else:
            # default DECT (existing)
            values = compute_ect(
                x,
                v=v,
                radius=radius,
                resolution=num_heights,
                scale=1,
                ect_fn=indicator,
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

        # Move directions to device; move data only for images (keep mesh objects untouched)
        directions = directions.cuda()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if data_type != "image" and data_type != "3d_mesh":
            raise NotImplementedError("DECT currently supports data_type='image' and '3d_mesh' only")

        if data_type == "image":
            # keep existing image path logic exactly as before
            data = data.cuda()
            img_complex = weighted_cubical(data)
            x = img_complex.dimensions[0][0]
            edges = img_complex.dimensions[1][0]
            triangles = img_complex.dimensions[2][0]
            simplices = (edges, triangles)

            H, W = data.shape[-2], data.shape[-1]
            radius = ((0.5 * H) ** 2 + (0.5 * W) ** 2) ** 0.5

        else:  # data_type == "3d_mesh"
            # mesh_to_complex loads vertices / edges / triangles already on CUDA device
            mesh_complex = mesh_to_complex(data, device=torch.device("cuda"))

            x = mesh_complex.dimensions[0][0]           # [N, 3]
            edge_index = mesh_complex.dimensions[1][0].T  # convert to [2, num_edges]
            face_index = mesh_complex.dimensions[2][0].T  # convert to [3, num_faces]

            simplices = (edge_index, face_index)

            # bounding-sphere radius of point cloud (same intent as CPU implementation)
            mins = x.min(dim=0).values
            maxs = x.max(dim=0).values
            center = 0.5 * (mins + maxs)
            radius = ( ((x - center) ** 2).sum(dim=1).sqrt().max().item() )

        end_event.record()
        torch.cuda.synchronize()
        result.complex_construction_time = start_event.elapsed_time(end_event) / 1000.0

        v = directions.T.contiguous()

        # warmup step: match function used in the timed compute below
        if data_type == "3d_mesh":
            for _ in range(10):
                _ = compute_ect_mesh(
                    x=x,
                    edge_index=edge_index,
                    face_index=face_index,
                    v=v,
                    radius=radius,
                    resolution=num_heights,
                    scale=1.0,
                    index=None,
                )
        else:
            for _ in range(10):
                _ = compute_ect(
                    x,
                    v=v,
                    radius=radius,
                    resolution=num_heights,
                    scale=1,
                    ect_fn=indicator
                )
        torch.cuda.synchronize()

        start_event.record()
        if data_type == "3d_mesh":
            values = compute_ect_mesh(
                x=x,
                edge_index=edge_index,
                face_index=face_index,
                v=v,
                radius=radius,
                resolution=num_heights,
                scale=1.0,
                index=None,
            )
        else:
            values = compute_ect(
                x,
                v=v,
                radius=radius,
                resolution=num_heights,
                scale=1,
                ect_fn=indicator,
            )
        end_event.record()
        torch.cuda.synchronize()

        result.computation_time = start_event.elapsed_time(end_event) / 1000.0
        result.vectorization_time = 0.0
        result.value = values
        return result


class DECT_WECT_MPS_Implementation(I_Implementation):

    def compute(self, data, directions: torch.Tensor, num_heights: int,
                cores: int = 1, data_type: str = "image") -> ImplementationResult:

        result = ImplementationResult()
        result.device_used = "mps"
        torch.set_grad_enabled(False)

        # move directions to MPS; move data only for images
        directions = directions.to("mps")

        # 1. Complex construction on MPS
        torch.mps.synchronize()
        t0 = time.perf_counter()

        if data_type != "image" and data_type != "3d_mesh":
            raise NotImplementedError("DECT currently supports data_type='image' and '3d_mesh' only")

        if data_type == "image":
            data = data.to("mps")
            img_complex = weighted_cubical(data)
            x = img_complex.dimensions[0][0]
            edges = img_complex.dimensions[1][0]
            triangles = img_complex.dimensions[2][0]
            simplices = (edges, triangles)

            H, W = data.shape[-2], data.shape[-1]
            radius = ((0.5 * H) ** 2 + (0.5 * W) ** 2) ** 0.5

        else:  # data_type == "3d_mesh"
            # mesh_to_complex on MPS device
            mesh_complex = mesh_to_complex(data, device=torch.device("mps"))

            x = mesh_complex.dimensions[0][0]           # [N, 3]
            edge_index = mesh_complex.dimensions[1][0].T  # convert to [2, num_edges]
            face_index = mesh_complex.dimensions[2][0].T  # convert to [3, num_faces]

            simplices = (edge_index, face_index)

            mins = x.min(dim=0).values
            maxs = x.max(dim=0).values
            center = 0.5 * (mins + maxs)
            radius = ( ((x - center) ** 2).sum(dim=1).sqrt().max().item() )

        torch.mps.synchronize()
        result.complex_construction_time = time.perf_counter() - t0

        v = directions.T.contiguous()

        # 2. Compute DECT
        torch.mps.synchronize()
        t1 = time.perf_counter()

        if data_type == "3d_mesh":
            values = compute_ect_mesh(
                x=x,
                edge_index=edge_index,
                face_index=face_index,
                v=v,
                radius=radius,
                resolution=num_heights,
                scale=1.0,
                index=None,
            )
        else:
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
