import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from typing import Callable

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


class GaussianProcess(nn.Module):
    def __init__(
        self,
        grid_min: float = -2,
        grid_max: float = 2,
        num_grids: int = 8,
        length_scale: float = 1,
        kernel_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.length_scale = length_scale
        self.grid = torch.linspace(grid_min, grid_max, num_grids)
        # 默认为 RBF Kernel
        self.kernel_function = kernel_function or self.rbf_kernel

    def rbf_kernel(self, x1, x2):
        # RBF kernel function (Gaussian kernel)
        dist = x1[..., None] - x2
        return torch.exp(-0.5 * (dist / self.length_scale) ** 2)

    # def rbf_kernel(self, x1, x2):
    #     # 计算 denominator，保持与 FastKANLayer 的 RBF 方式一致
    #     denominator = (self.grid_max - self.grid_min) / (self.num_grids - 1)
        # dist = x1[..., None] - x2
        # return torch.exp(-((dist / denominator) ** 2))

    def polynomial_kernel(self, x1, x2):
        # Polynomial kernel of degree 2
        dist = x1[..., None] - x2
        return (1 + dist / self.length_scale) ** 2

    def relu_kernel(self, x1, x2):
        # ReLU kernel function
        dist = x1[..., None] - x2
        return F.relu(1 - 0.5 * (dist / self.length_scale) ** 2)

    def tanh_kernel(self, x1, x2):
        # Ensure x1 and x2 are on the same device
        device = x1.device
        x2 = x2.to(device)
        # Tanh kernel function
        dist = x1[..., None] - x2
        return torch.tanh(1 - 0.5 * (dist / self.length_scale) ** 2)
    
    def rational_quadratic_kernel(self, x1, x2):
        # Rational quadratic kernel
        dist = x1[..., None] - x2
        return 1 / (1 + 0.5 * (dist / self.length_scale) ** 2)

    def matern_kernel(self, x1, x2, nu=0.5):
        # Matérn kernel with parameter nu (smoothness)
        if nu is None:
            nu = self.nu
        dist = torch.abs(x1[..., None] - x2)
        # sqrt_term = torch.sqrt(2 * nu) * dist / self.length_scale
        sqrt_term = torch.sqrt(torch.tensor(2 * nu, dtype=dist.dtype, device=dist.device)) * dist / self.length_scale

        if nu == 0.5:
            return torch.exp(-dist / self.length_scale)  # Special case of Matérn
        elif nu == 1.5:
            return (1 + sqrt_term) * torch.exp(-sqrt_term)
        elif nu == 2.5:
            return (1 + sqrt_term + (sqrt_term ** 2) / 3) * torch.exp(-sqrt_term)
        else:
            raise ValueError("nu should be 0.5, 1.5, or 2.5 for the Matérn kernel.")

    def exp_sine_squared_kernel(self, x1, x2):
        # Exp-Sine-Squared kernel
        dist = torch.abs(x1[..., None] - x2)
        return torch.exp(-2 * (torch.sin(math.pi * dist / self.length_scale) ** 2))

    def dot_product_kernel(self, x1, x2):
        # Dot product kernel
        return torch.matmul(x1.unsqueeze(-1), x2.unsqueeze(0)) + 1
    
    def tanh_matern(self, x, grid):
        dist = x[..., None] - grid
        tanh_output = self.tanh_kernel(x, grid)
        tanh_output = torch.clamp(tanh_output, min=0)  # 截取负数部分
        matern_output = self.matern_kernel(x, grid, nu=2.5)
        return 1.3 * tanh_output * matern_output

    def forward(self, x):
        # x: (batch_size, input_dim)
        self.grid = self.grid.to(x.device)
        # 这里让 GP 返回 (batch_size, input_dim, num_grids)
        # 相当于对 x 的每个维度都做了 num_grids 个 kernel base
        gp_out = []
        for i in range(x.size(-1)):  # 针对每个输入维度
            # x[..., i]: (batch_size,)
            # kernel_function(x[..., i], self.grid): (batch_size, num_grids)
            gp_out.append(self.kernel_function(x[..., i], self.grid))
        # 拼成 (batch_size, input_dim, num_grids)
        gp_out = torch.stack(gp_out, dim=1)
        return gp_out




class GPKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        # GP相关超参
        # grid_min: float = -1,
        # grid_max: float = 1,
        # num_grids: int = 16,
        # length_scale: float = 2,
        grid_min: float = -2,
        grid_max: float = 2,
        num_grids: int = 8,
        length_scale: float = 1,
        kernel_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        # 其它
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gp = GaussianProcess(
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            length_scale=length_scale,
            kernel_function=kernel_function,
        )
        # 为了拿到 (batch_size, input_dim, output_dim)，
        # 定义一个三维权重张量 gp_weight: (input_dim, num_grids, output_dim)
        # 这样可以通过爱因斯坦求和，把 GP 的 (b, i, n) 变成 (b, i, d)
        self.gp_weight = nn.Parameter(torch.empty(input_dim, num_grids, output_dim))
        nn.init.trunc_normal_(self.gp_weight, mean=0, std=init_scale)

    def forward(self, x: torch.Tensor):
        """
        我们希望最终 output 的形状为 (batch_size, input_dim, output_dim),
        然后做 output = base_output + spline_output + gp_output
        """
        # x: (batch_size, input_dim)

        # gp_features shape = (b, i, num_grids)
        gp_features = self.gp(x)
        # gp_weight shape = (i, num_grids, d)
        # 做 "b i n, i n d -> b i d"
        gp_output = torch.einsum("bin,ind->bid", gp_features, self.gp_weight)
        return gp_output

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
        gp_grid_min: float = -2,
        gp_grid_max: float = 2,
        gp_num_grids: int = 8,
        gp_length_scale: float = 1,
        gp_kernel_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        gp_spline_weight_init_scale: float = 0.1,
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

        # GPKANLayer
        self.gp_layer = GPKANLayer(
            input_dim=n_features,
            output_dim=d_embedding,
            grid_min=gp_grid_min,
            grid_max=gp_grid_max,
            num_grids=gp_num_grids,
            length_scale=gp_length_scale,
            # kernel_function=gp_kernel_function,
            # kernel_function=GaussianProcess().tanh_matern,
            # kernel_function=GaussianProcess().matern_kernel,
            kernel_function=GaussianProcess().rbf_kernel,
            init_scale=gp_spline_weight_init_scale,
        )

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
        # assert x.dim() == 2 and x.size(1) == self.n_features

        # 基础变换
        base_output = self.base_activation(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)

        # Spline 变换
        # spline_bases = self.b_splines(x)
        # spline_output = torch.einsum("bni, ndi -> bnd", spline_bases, self.scaled_spline_weight)

        # Fourier 变换
        # fourier_output = self.fourier_layer(x)  # (batch_size, n_features, d_embedding)

        # GP 变换
        gp_output = self.gp_layer(x)  # (batch_size, n_features, d_embedding)
        
        # 合并所有部分
        # output = base_output + spline_output + fourier_output + gp_output
        # output = base_output + spline_output 
        output = base_output + gp_output
        return output