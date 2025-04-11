import torch
import torch.nn as nn
import math
import numpy as np

import torch.nn.functional as F

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, n_features, d_embedding, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer, self).__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.gridsize = gridsize
        self.addbias = addbias

        # 平滑初始化，使得高频部分的幅度更小
        grid_norm_factor = (torch.arange(gridsize) + 1) ** 2 if smooth_initialization else math.sqrt(gridsize)

        # 归一化 Fourier 系数
        self.fourier_coeffs = nn.Parameter(torch.randn(2, n_features, d_embedding, gridsize) / (math.sqrt(n_features) * grid_norm_factor))

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, n_features, d_embedding))

    def forward(self, x):
        """
        x: (batch_size, n_features)
        Returns:
        y: (batch_size, n_features, d_embedding)
        """
        batch_size = x.shape[0]

        # 生成 Fourier 频率
        k = torch.arange(1, self.gridsize + 1, device=x.device).view(1, 1, 1, self.gridsize)  # (1, 1, 1, gridsize)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (batch_size, n_features, 1, 1)

        # 计算 Fourier 变换
        cos_part = torch.cos(k * x)  # (batch_size, n_features, 1, gridsize)
        sin_part = torch.sin(k * x)  # (batch_size, n_features, 1, gridsize)

        # 计算加权和
        y = torch.sum(cos_part * self.fourier_coeffs[0:1], dim=-1) + torch.sum(sin_part * self.fourier_coeffs[1:2], dim=-1)  # (batch_size, n_features, d_embedding)

        if self.addbias:
            y += self.bias

        return y

class KANEmbeddings(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        fourier_gridsize=10,  # Fourier 变换的 grid size
    ):
        super(KANEmbeddings, self).__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 初始化网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(n_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 线性变换 (Base Weight)
        self.base_weight = nn.Parameter(torch.Tensor(n_features, d_embedding))
        self.spline_weight = nn.Parameter(torch.Tensor(n_features, d_embedding, grid_size + spline_order))

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(n_features, d_embedding))

        # Fourier KAN Layer
        self.fourier_layer = NaiveFourierKANLayer(n_features, d_embedding, gridsize=fourier_gridsize)

        # 参数
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.n_features, self.d_embedding) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
        if self.enable_standalone_scale_spline:
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.n_features

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.n_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.n_features
        assert y.size() == (x.size(0), self.n_features, self.d_embedding)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(0, 2, 1)

        assert result.size() == (self.n_features, self.d_embedding, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.n_features

        # 基础变换
        base_output = self.base_activation(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)

        # Spline 变换
        spline_bases = self.b_splines(x)
        spline_output = torch.einsum("bni, ndi -> bnd", spline_bases, self.scaled_spline_weight)

        # Fourier 变换
        fourier_output = self.fourier_layer(x)  # (batch_size, n_features, d_embedding)

        # 合并所有部分
        output = base_output + spline_output + fourier_output
        return output