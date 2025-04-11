import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 定义数据集文件夹和算法对应的日志文件名关键字
datasets = ['PoPu†', 'PoPu‡', 'Pmat†', 'Pmat‡', 'SLP']
algorithms = {
    'KAFT': 'effkan',
    'NativeFT': 'fttransformer',
    'MKAFT(ours)': 'matern'
}

# 正则表达式提取数据
epoch_pattern = r'Epoch (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+), Test Loss: ([\d.]+), Val Acc: ([\d.]+), Test Acc: ([\d.]+)'

# 创建3行5列的大图，调整行间距
fig, axes = plt.subplots(3, 5, figsize=(25, 10))  # 增加figsize的高度
# fig.suptitle(' Comparison of FT-Transformer with Native, KAN-Integrated, and Matern-driven KAN-Integrated Feature Tokenizers', fontsize=16)

# 调整行间距
plt.subplots_adjust(hspace=0.9)  # 增加hspace的值，调整行距

# 遍历每个数据集
for col, dataset in enumerate(datasets):
    # 在每列的最顶端子图添加数据集标题
    axes[0, col].set_title(dataset, fontsize=14, pad=20)  # 设置标题，并增加pad以避免标题与图重叠
    
    # 遍历每个算法
    for algorithm, keyword in algorithms.items():
        # 查找对应的日志文件
        log_file = None
        for file in os.listdir(dataset):
            if keyword in file:
                log_file = os.path.join(dataset, file)
                break
        
        if not log_file:
            print(f"Log file for {algorithm} in {dataset} not found!")
            continue
        
        # 读取日志文件
        with open(log_file, 'r') as file:
            log_data = file.read()
        
        # 解析数据
        matches = re.findall(epoch_pattern, log_data)
        epochs = []
        train_losses = []
        test_losses = []
        test_accs = []
        
        for match in matches:
            epochs.append(int(match[0]))
            train_losses.append(float(match[1]))
            test_losses.append(float(match[3]))
            test_accs.append(float(match[5]))
        
        # 第一行：epoch vs loss
        axes[0, col].plot(epochs, train_losses, label=f'{algorithm} Train Loss', marker='o', linestyle='-', markersize=4)  # 调整marker大小
        axes[0, col].plot(epochs, test_losses, label=f'{algorithm} Test Loss', marker='x', linestyle='--', markersize=4)  # 调整marker大小
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Loss')
        axes[0, col].legend()
        axes[0, col].grid(True)
        
        # 第二行：train loss vs test loss
        # 对 Train Loss 做降序排序，同时保持 Test Loss 对应关系
        sorted_indices = np.argsort(train_losses)[::-1]  # 获取降序排序的索引
        sorted_train_losses = np.array(train_losses)[sorted_indices]
        sorted_test_losses = np.array(test_losses)[sorted_indices]

        axes[1, col].scatter(sorted_train_losses, sorted_test_losses, label=algorithm, alpha=0.5, s=20)
        axes[1, col].invert_xaxis()  # 让 x 轴真正从大到小显示
        axes[1, col].set_xlabel('Train Loss (Descending)')
        axes[1, col].set_ylabel('Test Loss')
        axes[1, col].legend()
        axes[1, col].grid(True)
        
        # 第三行：epoch vs test accuracy
        axes[2, col].plot(epochs, test_accs, label=f'{algorithm}', marker='o', linestyle='-', markersize=4)  # 调整marker大小
        axes[2, col].set_xlabel('Epoch')
        axes[2, col].set_ylabel('Test Accuracy')
        axes[2, col].legend()
        axes[2, col].grid(True)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("loss_of_MKAFT_KAFT_FT_3X5.png", dpi=1000, format='png', bbox_inches='tight')
plt.show()