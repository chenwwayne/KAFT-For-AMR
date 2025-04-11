import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from scipy.special import kv  # Modified Bessel function of the second kind

class MaternKernel(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        nu: float = 1.5,  # Smoothness parameter
        lengthscale: float = 1.0,
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.nu = nu
        self.lengthscale = lengthscale
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Expand for broadcasting
        dists = torch.abs(x - self.grid) / self.lengthscale
        if self.nu == 0.5:
            basis = torch.exp(-dists)
        elif self.nu == 1.5:
            basis = (1 + math.sqrt(3) * dists) * torch.exp(-math.sqrt(3) * dists)
        elif self.nu == 2.5:
            basis = (1 + math.sqrt(5) * dists + 5 / 3 * dists ** 2) * torch.exp(-math.sqrt(5) * dists)
        else:
            raise ValueError("Only nu=0.5, 1.5, or 2.5 are supported for Matérn kernel.")
        return basis

class MaternKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        nu: float = 1.5,
        lengthscale: float = 1.0,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.matern = MaternKernel(grid_min, grid_max, num_grids, nu, lengthscale)
        self.spline_linear = nn.Linear(input_dim * num_grids, output_dim, bias=False)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            matern_basis = self.matern(self.layernorm(x))
        else:
            matern_basis = self.matern(x)
        ret = self.spline_linear(matern_basis.view(*matern_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret
