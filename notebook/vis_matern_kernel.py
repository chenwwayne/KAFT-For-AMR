import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 创建数据
X = np.linspace(0, 10, 10).reshape(-1, 1)  # 输入数据，减少数据点以简化示例
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # 添加噪声的目标值

# 定义 Matérn 核函数
kernel = Matern(length_scale=1.0, nu=1.5)

# 初始化高斯过程回归模型
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 拟合模型
gp.fit(X, y)

# 生成后验分布样本
X_prior = np.linspace(0, 10, 1000).reshape(-1, 1)
n_samples = 5
y_posterior_samples = gp.sample_y(X_prior, n_samples=n_samples)

# 预测后验分布的均值和标准差
y_mean, y_std = gp.predict(X_prior, return_std=True)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制后验分布图
for i, y_samples in enumerate(y_posterior_samples.T):  # 转置以匹配X_prior的形状
    ax.plot(X_prior, y_samples, linestyle='--', label=f'Sampled function #{i+1}')
ax.plot(X_prior, y_mean, 'k-', label='Mean')
ax.fill_between(X_prior.ravel(), y_mean - y_std, y_mean + y_std, color='gray', alpha=0.2, label='± 1 std. dev.')
ax.scatter(X, y, color='red', label='Observations')  # 添加观测数据点
ax.set_title('Posterior Distribution with Matérn Kernel', fontsize=32)  # 修改标题字体大小
ax.set_xlabel('x', fontsize=16)  # 修改x轴标签字体大小
ax.set_ylabel('y', fontsize=16)  # 修改y轴标签字体大小
ax.legend(fontsize=15)  # 修改图例字体大小

# 调整布局
plt.tight_layout()
# 保存图形为PNG文件，dpi设置为300
plt.savefig('matern_kernel_posterior.png', dpi=300)
plt.show()