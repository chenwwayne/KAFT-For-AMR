import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 调整数据集的顺序
datasets = ['PoPu†', 'PoPu‡', 'Pmat†', 'Pmat‡', 'SLP']
algorithms = {
    'KAFT-T': 'effkan',
    'FT-T': 'fttransformer',
    'MKAFT-T(ours)': 'matern'
}

# 定义颜色
colors = {
    'MKAFT-T(ours)': {
        'train': 'darkorange',  # MKAFT Train Loss 使用深橙色
        'test': 'gold'        # MKAFT Test Loss 使用浅橙色
    },
    'KAFT-T': {
        'train': 'blue',        # KAFT Train Loss 使用蓝色
        'test': 'lightblue'     # KAFT Test Loss 使用浅蓝色
    },
    'FT-T': {
        'train': 'green',       # NativeFT Train Loss 使用绿色
        'test': 'lightgreen'    # NativeFT Test Loss 使用浅绿色
    }
}

# 正则表达式提取数据
epoch_pattern = r'Epoch (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+), Test Loss: ([\d.]+), Val Acc: ([\d.]+), Test Acc: ([\d.]+)'

# 创建 5 行 × 3 列 的大图，并设置合适的 fig 大小
fig, axes = plt.subplots(5, 3, figsize=(14, 13))
# 可再适当调 left= 值，让所有子图适当向右移动
plt.subplots_adjust(left=0.1, hspace=0.8)

for row, dataset in enumerate(datasets):
    # 在最左侧添加 逆时针旋转 90 度 的数据集标题
    # x=-0.25 表示比 -0.15 更远离第一列子图（更靠左），可根据需求再微调
    axes[row, 0].text(
        -0.25, 0.5, dataset,
        rotation=0,        # 逆时针旋转 90 度
        fontsize=14,
        transform=axes[row, 0].transAxes,
        ha='center', va='center'
    )
    
    # 遍历每个算法
    for algorithm, keyword in algorithms.items():
        log_file = None
        if os.path.exists(dataset):
            for file in os.listdir(dataset):
                if keyword in file:
                    log_file = os.path.join(dataset, file)
                    break
        
        if not log_file:
            print(f"Log file for {algorithm} in {dataset} not found!")
            continue
        
        with open(log_file, 'r') as file:
            log_data = file.read()
        
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
        
        #
        # 第一列：epoch vs. (train loss & test loss)
        #
        axes[row, 0].plot(
            epochs, train_losses, 
            label=f'{algorithm} Train Loss', 
            marker='o', linestyle='-', markersize=4,
            color=colors[algorithm]['train']  # 使用定义的颜色
        )
        axes[row, 0].plot(
            epochs, test_losses, 
            label=f'{algorithm} Test Loss', 
            marker='x', linestyle='--', markersize=4,
            color=colors[algorithm]['test']  # 使用定义的颜色
        )
        axes[row, 0].set_xlabel('Epoch')
        axes[row, 0].set_ylabel('Loss')
        axes[row, 0].legend()
        axes[row, 0].grid(True)
        
        #
        # 第二列：train loss vs. test loss（降序可视化 & 反转x轴）
        #
        sorted_indices = np.argsort(train_losses)[::-1]  
        sorted_train_losses = np.array(train_losses)[sorted_indices]
        sorted_test_losses = np.array(test_losses)[sorted_indices]

        axes[row, 1].scatter(
            sorted_train_losses, sorted_test_losses, 
            label=algorithm, alpha=0.5, s=20,
            color=colors[algorithm]['train']  # 使用定义的颜色
        )
        axes[row, 1].invert_xaxis()
        axes[row, 1].set_xlabel('Train Loss (Descending)')
        axes[row, 1].set_ylabel('Test Loss')
        axes[row, 1].legend()
        axes[row, 1].grid(True)
        
        #
        # 第三列：epoch vs. test accuracy
        #
        axes[row, 2].plot(
            epochs, test_accs, 
            label=f'{algorithm}', 
            marker='o', linestyle='-', markersize=4,
            color=colors[algorithm]['train']  # 使用定义的颜色
        )
        axes[row, 2].set_xlabel('Epoch')
        axes[row, 2].set_ylabel('Test Accuracy')
        axes[row, 2].legend()
        axes[row, 2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("loss_of_MKAFT_KAFT_FT_5X3.png", dpi=300, format='png', bbox_inches='tight')
fig.savefig("loss_of_MKAFT_KAFT_FT_5X3.svg", dpi=300, format='svg', bbox_inches='tight')
plt.show()