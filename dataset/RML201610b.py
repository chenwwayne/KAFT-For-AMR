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
    file = open(filename, 'rb')
    Xd = pickle.load(file, encoding='bytes')
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]
    print("Modulations:", mods)
    print("SNRs:", snrs)

    X = []
    lbl = []
    train_idx = []
    val_idx = []  # 验证集索引
    test_idx = []  # 测试集索引（将与验证集相同）

    np.random.seed(2023)
    global_idx = 0

    # 仅修改这两个比例系数（其他所有代码保持不变）
    train_ratio = 0.8  # 原0.6改为0.8
    val_ratio = 0.2    # 原0.2改为0.2（测试集将复用这部分数据）

    for mod in mods:
        for snr in snrs:
            block_data = Xd[(mod, snr)]
            block_len = block_data.shape[0]
            
            X.append(block_data)
            lbl.extend([(mod, snr)] * block_len)
            
            # 生成当前块的索引并随机打乱
            block_indices = list(range(global_idx, global_idx + block_len))
            np.random.shuffle(block_indices)
            
            # 计算拆分点（仅修改比例，逻辑不变）
            train_end = int(train_ratio * block_len)
            val_end = train_end + int(val_ratio * block_len)
            
            # 分配索引（保持原逻辑）
            train_idx.extend(block_indices[:train_end])
            val_idx.extend(block_indices[train_end:val_end])
            # 使测试集使用与验证集相同的数据
            test_idx = val_idx.copy()  # 关键修改：让测试集索引与验证集相同
            
            global_idx += block_len

    # 以下所有代码保持不变...
    X = np.vstack(X)
    print(f"Total samples: {X.shape[0]}")

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]  # 这将与X_val相同

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)], dtype=np.float32)
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    mod_class_per_sample = [mods.index(pair[0]) for pair in lbl]
    Y_train = to_onehot([mod_class_per_sample[i] for i in train_idx])
    Y_val = to_onehot([mod_class_per_sample[i] for i in val_idx])
    Y_test = Y_val.copy()  # 测试集标签与验证集相同

    X_train, X_val, X_test = to_amp_phase(X_train, X_val, X_test, maxlen)
    
    X_train = X_train[:, :maxlen, :]
    X_val = X_val[:, :maxlen, :]
    X_test = X_test[:, :maxlen, :]
    
    X_train = norm_pad_zeros(X_train, maxlen)
    X_val = norm_pad_zeros(X_val, maxlen)
    X_test = norm_pad_zeros(X_test, maxlen)

    print("\nFinal data shapes:")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")  # 将显示与验证集相同的数据

    file.close()
    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx)

if __name__ == '__main__':
    # 使用示例
    data_path = "RML2016.10b.dat"  # 修改为您的实际路径
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), _ = load_data(data_path)