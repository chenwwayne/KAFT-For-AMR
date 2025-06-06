
import numpy as np
import pickle
from numpy import linalg as la

maxlen = 128  # 定义信号长度

def l2_normalize(x, axis=-1):
    """L2归一化处理"""
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)

def norm_pad_zeros(X, nsamples):
    """归一化并补零处理"""
    print("Padding:", X.shape)
    for i in range(X.shape[0]):
        X[i, :, 0] = X[i, :, 0] / la.norm(X[i, :, 0], 2)
    return X

def to_amp_phase(X_train, X_val, X_test, nsamples):
    """将IQ信号转换为幅度和相位"""
    # 训练集转换
    X_train_cmplx = X_train[:, 0, :] + 1j * X_train[:, 1, :]
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, 1, :], X_train[:, 0, :]) / np.pi
    X_train = np.stack([X_train_amp, X_train_ang], axis=1).transpose(0, 2, 1)

    # 验证集转换
    X_val_cmplx = X_val[:, 0, :] + 1j * X_val[:, 1, :]
    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:, 1, :], X_val[:, 0, :]) / np.pi
    X_val = np.stack([X_val_amp, X_val_ang], axis=1).transpose(0, 2, 1)

    # 测试集转换
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi
    X_test = np.stack([X_test_amp, X_test_ang], axis=1).transpose(0, 2, 1)

    return X_train, X_val, X_test

def load_data(filename):
    """加载并预处理数据集"""
    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
    print("Modulations:", mods)
    print("SNRs:", snrs)

    X = []
    lbl = []
    train_idx = []
    val_test_idx = []  # 合并验证和测试集索引

    np.random.seed(2023)
    global_idx = 0

    # 定义数据集拆分比例
    train_ratio = 0.8  # 80%训练集
    val_test_ratio = 0.2  # 20%验证/测试集

    for mod in mods:
        for snr in snrs:
            block_data = Xd[(mod, snr)]
            block_len = block_data.shape[0]
            
            X.append(block_data)
            lbl.extend([(mod, snr)] * block_len)
            
            # 生成当前块的索引并随机打乱
            block_indices = list(range(global_idx, global_idx + block_len))
            np.random.shuffle(block_indices)
            
            # 计算拆分点
            train_end = int(train_ratio * block_len)
            
            # 分配索引
            train_idx.extend(block_indices[:train_end])
            val_test_idx.extend(block_indices[train_end:])  # 剩余的20%用于验证和测试
            
            global_idx += block_len

    # 合并所有数据
    X = np.vstack(X)
    print(f"Total samples: {X.shape[0]}")

    # 提取数据
    X_train = X[train_idx]
    X_val_test = X[val_test_idx]  # 验证和测试使用相同数据

    # 转换为one-hot标签
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)], dtype=np.float32)
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    mod_class_per_sample = [mods.index(pair[0]) for pair in lbl]
    Y_train = to_onehot([mod_class_per_sample[i] for i in train_idx])
    Y_val_test = to_onehot([mod_class_per_sample[i] for i in val_test_idx])

    # 数据预处理流程
    X_train, X_val, X_test = to_amp_phase(X_train, X_val_test, X_val_test, maxlen)  # 注意这里验证和测试使用相同数据
    
    X_train = X_train[:, :maxlen, :]
    X_val = X_val[:, :maxlen, :]
    X_test = X_test[:, :maxlen, :]
    
    X_train = norm_pad_zeros(X_train, maxlen)
    X_val = norm_pad_zeros(X_val, maxlen)
    X_test = norm_pad_zeros(X_test, maxlen)

    print("\nFinal data shapes:")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_val: {X_val.shape}, Y_val: {Y_val_test.shape}")  # 验证集标签
    print(f"X_test: {X_test.shape}, Y_test: {Y_val_test.shape}")  # 测试集标签

    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val_test), (X_test, Y_val_test), (train_idx, val_test_idx, val_test_idx)

if __name__ == '__main__':
    # 使用示例
    data_path = "RML201604c.pkl"  # 修改为您的实际路径
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), _ = load_data(data_path)