import torch

from domain.experiment import ExperimentParameters
from domain.implementation import ImplementationResult

class I_Implementation():
    def __init__(self):
        pass

    def compute(self, data, directions: torch.Tensor, num_heights: int, cores: int) -> ImplementationResult:
        pass
