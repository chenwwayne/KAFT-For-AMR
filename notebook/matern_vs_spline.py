# -*- coding: utf-8 -*-
import torch
import math
import matplotlib.pyplot as plt

def matern_kernel(x, grid, length_scale=1.0, nu=1.5):
    """
    计算一维 Matérn 核函数。
    x: [N]  (输入数据)
    grid: [M] (核的参考网格点)
    length_scale: R+ (长度尺度)
    nu: 实数 (通常为 0.5, 1.5, 或 2.5)
    返回形状: [N, M]
    """
    device = x.device
    x = x.to(device)
    grid = grid.to(device)

    dist = torch.abs(x.unsqueeze(-1) - grid)

    if nu == 0.5:
        return torch.exp(-dist / length_scale)
    elif nu == 1.5:
        sqrt3 = math.sqrt(3)
        tmp = sqrt3 * dist / length_scale
        return (1.0 + tmp) * torch.exp(-tmp)
    elif nu == 2.5:
        sqrt5 = math.sqrt(5)
        tmp = sqrt5 * dist / length_scale
        return (1.0 + tmp + (tmp**2) / 3.0) * torch.exp(-tmp)
    else:
        raise ValueError("nu should be one of {0.5, 1.5, 2.5} in this example.")

def b_spline_bases(x, grid, order=3):
    """
    计算一维的 B-spline 基函数 (以非严格方式演示)。
    x: [N]  (输入数据)
    grid: [G+1] (网格划分, 假设需要 G 个区间)
    order: B-spline 的阶数 (默认三次样条)
    
    返回形状: [N, G + order]
    """
    
    device = x.device
    x = x.to(device)
    grid = grid.to(device)

    G = grid.shape[0] - 1  
    num_bases = G + order
    
    bases = torch.zeros((x.shape[0], num_bases), device=device)
    
    if order == 1:
        for i in range(G):
            left = grid[i]
            right = grid[i+1]
            mask = (x >= left) & (x < right)
            bases[mask, i] = 1.0
            
        bases[x == grid[-1], G-1] = 1.0
    
    else:
        raise NotImplementedError(
            "示例中仅演示 order=1 的分段线性插值；若需三次样条，请自行扩展。"
        )
    
    return bases

if __name__ == "__main__":
    # ========== 1. 生成测试数据 (一维) ==========
    x = torch.linspace(-2, 2, steps=100)  # 细一点用于画连续曲线
    
    # ========== 2. Matérn Kernel ==========
    gp_grid = torch.linspace(-2, 2, steps=6)  # 核参考网格点
    matern_out = matern_kernel(x, gp_grid, length_scale=1.0, nu=1.5)
    # matern_out.shape => [100, 6]
    
    # ========== 3. B-spline ==========
    bspline_grid = torch.linspace(-2, 2, steps=5)  # 4 区间 => G=4 => G+1=5
    bspline_out = b_spline_bases(x, bspline_grid, order=1)
    # bspline_out.shape => [100, 5]

    # ========== 4. 可视化 ==========

    # --- (A) Matérn Kernel ---
    plt.figure(figsize=(10, 4))
    for i in range(matern_out.shape[1]):
        plt.plot(x.cpu().numpy(), matern_out[:, i].cpu().numpy(), label=f"k(x,z_{i})")
    plt.title("Matérn Kernel (nu=1.5) with 6 grid points")
    plt.xlabel("x")
    plt.ylabel("kernel value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- (B) B-spline ---
    plt.figure(figsize=(10, 4))
    for j in range(bspline_out.shape[1]):
        plt.plot(x.cpu().numpy(), bspline_out[:, j].cpu().numpy(), label=f"B-spline {j}")
    plt.title("B-spline bases (order=1) with G=4")
    plt.xlabel("x")
    plt.ylabel("basis value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()