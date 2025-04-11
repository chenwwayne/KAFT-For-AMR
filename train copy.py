import math
import warnings
from typing import Dict, Tuple

import delu  # Deep Learning Utilities
import numpy as np
import time
import torch
import os
import gc
import argparse
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
from thop import profile, clever_format
import sklearn.metrics
from typing import Callable, Tuple, Dict

from rtdl_revisiting_models import FTTransformer

warnings.simplefilter("ignore")
warnings.resetwarnings()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

delu.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def reshape_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将数据展平为 (n_samples, 256)"""
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    return X_train_reshaped, X_val_reshaped, X_test_reshaped

def load_and_preprocess_data(load_data_fn: Callable[[str], Tuple], data_path: str, batch_size=1, data_ratio=1.0):
    """
    加载并预处理 RML2016.10a 数据集，使用 One-hot 编码。
    额外返回 (mods, snrs, lbl, test_idx) 方便后续在 evaluate_by_snr 使用。
    
    参数:
        data_path: 数据集路径
        data_ratio: 要使用的数据比例 (0.0-1.0)
    """
    # 第五个返回值是 (train_idx, val_idx, test_idx) 的元组
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
        load_data_fn(data_path)

    # 按比例截取数据
    if data_ratio < 1.0:
        train_size = int(len(X_train) * data_ratio)
        val_size = int(len(X_val) * data_ratio)
        test_size = int(len(X_test) * data_ratio)
        
        X_train = X_train[:train_size]
        Y_train = Y_train[:train_size]
        train_idx = train_idx[:train_size]
        
        X_val = X_val[:val_size]
        Y_val = Y_val[:val_size]
        val_idx = val_idx[:val_size]
        
        X_test = X_test[:test_size]
        Y_test = Y_test[:test_size]
        test_idx = test_idx[:test_size]

    # 展平
    X_train_reshaped, X_val_reshaped, X_test_reshaped = reshape_data(X_train, X_val, X_test)

    print("After reshaping and sampling:")
    print(f"X_train: {X_train_reshaped.shape} (sampled from original)")
    print(f"X_val:   {X_val_reshaped.shape} (sampled from original)")
    print(f"X_test:  {X_test_reshaped.shape} (sampled from original)")
    print(f"Y_train: {Y_train.shape}")

    n_classes = Y_train.shape[1]

    data = {
        "train": {
            "x": torch.as_tensor(X_train_reshaped, dtype=torch.float32),
            "y": torch.as_tensor(np.argmax(Y_train, axis=1), dtype=torch.long)
        },
        "val": {
            "x": torch.as_tensor(X_val_reshaped, dtype=torch.float32),
            "y": torch.as_tensor(np.argmax(Y_val, axis=1), dtype=torch.long)
        },
        "test": {
            "x": torch.as_tensor(X_test_reshaped, dtype=torch.float32),
            "y": torch.as_tensor(np.argmax(Y_test, axis=1), dtype=torch.long)
        }
    }
    n_cont_features = X_train_reshaped.shape[1]  # 256

    return data, n_classes, n_cont_features, mods, snrs, lbl, test_idx


def apply_model(model, batch: Dict[str, Tensor]) -> Tensor:
    """应用模型"""
    x = batch["x"].to(device)
    output = model(x)
    return output

def define_model(n_cont_features: int, n_classes: int, args):
    """定义模型"""
    model = FTTransformer(
        n_cont_features=n_cont_features,  # 256
        d_out=n_classes,
        **FTTransformer.get_default_kwargs(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 使用传入的学习率
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',       # 监控验证集准确率
        factor= args.lr_factor,
        patience= args.lr_patience,
        verbose=True
    )
    
    return model, optimizer, scheduler  # 现在返回三个对象

# def train_model(model, optimizer, scheduler, data, n_epochs, patience, batch_size, accumulation_steps=1, lr=1e-3):
#     """训练模型"""
#     loss_fn = torch.nn.CrossEntropyLoss()  # 使用多分类损失函数
#     scaler = GradScaler()

#     epoch_size = math.ceil(len(data["train"]["y"]) / batch_size)
#     timer = delu.tools.Timer()
#     early_stopping = delu.tools.EarlyStopping(patience, mode="max")
#     best = {"val": -math.inf, "test": -math.inf, "epoch": -1}

#     start_time = time.time()
#     best_time = None

#     print(f"Device: {device.type.upper()}")
#     print("-" * 88 + "\n")
#     timer.run()

#     optimizer.zero_grad()

#     for epoch in range(n_epochs):
#         train_loss = 0.0
#         model.train()

#         for i, batch_cpu in enumerate(tqdm(delu.iter_batches(data["train"], batch_size, shuffle=True), desc=f"Epoch {epoch}", total=epoch_size)):
#             batch = {k: v.to(device) for k, v in batch_cpu.items()}

#             with autocast('cuda'):
#                 output = apply_model(model, batch)
#                 target = batch["y"]
#                 loss = loss_fn(output, target) / accumulation_steps

#             scaler.scale(loss).backward()

#             if (i + 1) % accumulation_steps == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#             train_loss += loss.item() * len(batch["y"]) * accumulation_steps

#         train_loss /= len(data["train"]["y"])

#         model.eval()
#         val_loss = 0.0
#         test_loss = 0.0
#         with torch.no_grad(), autocast('cuda'):
#             for part in ["val", "test"]:
#                 loss_sum = 0.0
#                 for batch_cpu in delu.iter_batches(data[part], batch_size):
#                     batch = {k: v.to(device) for k, v in batch_cpu.items()}
#                     output = apply_model(model, batch)
#                     target = batch["y"]
#                     loss = loss_fn(output, target)
#                     loss_sum += loss.item() * len(batch["y"])
#                 if part == "val":
#                     val_loss = loss_sum / len(data[part]["y"])
#                 elif part == "test":
#                     test_loss = loss_sum / len(data[part]["y"])

#         val_score = evaluate(model, data, "val", batch_size=batch_size)[0]  # 只取 score
#         test_score = evaluate(model, data, "test", batch_size=batch_size)[0]  # 只取 score

#         scheduler.step(val_score)

#         print(
#             f"Epoch {epoch}, "
#             f"Train Loss: {train_loss:.4f}, "
#             f"Val Loss: {val_loss:.4f}, "
#             f"Test Loss: {test_loss:.4f}, "
#             f"Val Acc: {val_score:.4f}, "
#             f"Test Acc: {test_score:.4f}"
#         )

#         early_stopping.update(val_score)
#         if early_stopping.should_stop():
#             break

#         if val_score > best["val"]:
#             print("🌸 New best epoch! 🌸")
#             best = {"val": val_score, "test": test_score, "epoch": epoch}
#             best_time = time.time()

#         print()

#         if best_time is not None:
#             total_time = best_time - start_time
#         print(f"Time to best model: {total_time:.2f} seconds")
#     else:
#         print("No best model found.")

#     return best

def train_model(model, optimizer, scheduler, data, n_epochs, patience, batch_size, accumulation_steps=1):
    """训练模型"""
    loss_fn = torch.nn.CrossEntropyLoss()  # 使用多分类损失函数
    scaler = GradScaler()

    epoch_size = math.ceil(len(data["train"]["y"]) / batch_size)
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best = {"val": -math.inf, "test": -math.inf, "epoch": -1}

    start_time = time.time()
    best_time = None

    print(f"Device: {device.type.upper()}")
    print("-" * 88 + "\n")
    timer.run()

    optimizer.zero_grad()

    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()

        for i, batch_cpu in enumerate(tqdm(delu.iter_batches(data["train"], batch_size, shuffle=True), desc=f"Epoch {epoch}", total=epoch_size)):
            batch = {k: v.to(device) for k, v in batch_cpu.items()}

            with autocast('cuda'):
                output = apply_model(model, batch)
                # target = batch["y"].argmax(dim=1)  # 将独热编码转换为类别索引
                target = batch["y"]
                loss = loss_fn(output, target) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * len(batch["y"]) * accumulation_steps

        train_loss /= len(data["train"]["y"])

        model.eval()
        val_loss = 0.0
        test_loss = 0.0
        with torch.no_grad(), autocast('cuda'):
            for part in ["val", "test"]:
                loss_sum = 0.0
                for batch_cpu in delu.iter_batches(data[part], batch_size):
                    batch = {k: v.to(device) for k, v in batch_cpu.items()}
                    output = apply_model(model, batch)
                    # target = batch["y"].argmax(dim=1)  # 将独热编码转换为类别索引
                    target = batch["y"]
                    loss = loss_fn(output, target)
                    loss_sum += loss.item() * len(batch["y"])
                if part == "val":
                    val_loss = loss_sum / len(data[part]["y"])
                elif part == "test":
                    test_loss = loss_sum / len(data[part]["y"])

        val_score = evaluate(model, data, "val", batch_size=batch_size)[0]  # 只取 score
        test_score = evaluate(model, data, "test", batch_size=batch_size)[0]  # 只取 score

        # # 更新学习率
        # if isinstance(scheduler, ReduceLROnPlateau):
        #     scheduler.step(val_loss)
        # else:
        #     scheduler.step(val_score)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Val Acc: {val_score:.4f}, "
            f"Test Acc: {test_score:.4f}"
        )

        early_stopping.update(val_score)
        if early_stopping.should_stop():
            break

        if val_score > best["val"]:
            print("🌸 New best epoch! 🌸")
            best = {"val": val_score, "test": test_score, "epoch": epoch}
            best_time = time.time()

        print()

    if best_time is not None:
        total_time = best_time - start_time
        print(f"Time to best model: {total_time:.2f} seconds")
    else:
        print("No best model found.")

    return best


@torch.no_grad()
def evaluate(model, data, part: str, batch_size=32) -> Tuple[float, float, float]:
    model.eval()
    eval_batch_size = batch_size

    y_pred = []
    y_true = []
    total_samples = len(data[part]["y"])
    total_time = 0.0  # 总推理时间（秒）
    latency_list = []  # 记录每个 batch 的延迟

    for batch_cpu in delu.iter_batches(data[part], eval_batch_size):
        batch = {k: v.to(device) for k, v in batch_cpu.items()}

        # 预热（避免首次推理的 CUDA 初始化影响延迟）
        if len(latency_list) == 0:
            with autocast('cuda'):
                _ = apply_model(model, batch)

        # 正式测量时间
        start_time = time.time()
        with autocast('cuda'):
            output = apply_model(model, batch)
        torch.cuda.synchronize()  # 确保 CUDA 操作完成
        end_time = time.time()

        batch_time = end_time - start_time
        latency_list.append(batch_time)
        total_time += batch_time

        y_pred.append(output.cpu().numpy())
        y_true.append(batch["y"].cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = y_true
    score = sklearn.metrics.accuracy_score(y_true_classes, y_pred_classes)

    # 计算吞吐量（样本/秒）
    throughput = total_samples / total_time if total_time > 0 else 0.0

    # 计算平均延迟（秒/样本）
    avg_latency_per_sample = total_time / total_samples if total_samples > 0 else 0.0

    # 计算平均延迟（秒/batch）
    avg_latency_per_batch = np.mean(latency_list) if latency_list else 0.0

    return score, throughput, avg_latency_per_sample


@torch.no_grad()
def evaluate_by_snr(
    model,
    data: Dict[str, Dict[str, torch.Tensor]],
    lbl: list,
    snrs: list,
    test_idx: list,
    classes: list,
    batch_size=32
):
    model.eval()

    test_x = data["test"]["x"]
    test_y = data["test"]["y"].cpu().numpy()  # 确保是 numpy 数组
    test_y_classes = np.argmax(test_y, axis=1)  # 转换为类别索引

    test_y_pred = []

    n_samples_test = test_x.size(0)
    with autocast('cuda'):
        outputs = []
        for batch_cpu in delu.iter_batches({"x": test_x, "y": test_y}, batch_size=batch_size, shuffle=False):
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            output = apply_model(model, batch)
            outputs.append(output.cpu().numpy())
        y_pred_np = np.concatenate(outputs, axis=0)

    y_pred_classes = np.argmax(y_pred_np, axis=1)

    acc_per_snr = {}

    for snr in snrs:
        indices_of_snr = [
            i for i, global_idx in enumerate(test_idx)
            if lbl[global_idx][1] == snr
        ]
        pred_snr = y_pred_classes[indices_of_snr]
        true_snr = test_y_classes[indices_of_snr]
        if len(pred_snr) == 0:
            acc_per_snr[snr] = None
            continue
        correct = np.sum(pred_snr == true_snr)
        total = len(pred_snr)
        accuracy_snr = correct / total
        acc_per_snr[snr] = accuracy_snr

    print("\n===== Accuracy By SNR =====")
    for snr in sorted(acc_per_snr.keys()):
        acc_val = acc_per_snr[snr]
        if acc_val is None:
            print(f"SNR = {snr:>3}: No samples in test set.")
        else:
            print(f"SNR = {snr:>3}: Acc = {acc_val: .4f}")
    print("==========================\n")

    return acc_per_snr


def get_dataset_loader(dataset_version):
    """根据数据集版本返回对应的加载函数"""
    if dataset_version == '10a':
        from dataset.RML201610a import load_data
        return load_data, 'RML2016.10a.pkl'
    elif dataset_version == '10b':
        from dataset.RML201610b import load_data
        return load_data, 'RML2016.10b.dat'
    elif dataset_version == '04c':
        from dataset.RML201604c import load_data
        return load_data, 'RML2016.04c.pkl'


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='10b',
                        choices=['10a', '10b', '04c'], help='Dataset version')
    parser.add_argument('--data_path', type=str, default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--n_epochs', type=int, default=1_000_000_000)
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--lr_patience', type=int, default=5, help='学习率调度器的耐心值(多少个epoch无改善后降学习率)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='学习率衰减系数')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help='运行设备')
    parser.add_argument('--gpu', type=str, default="0", help='指定要使用的 GPU 设备编号，如 "0" 或 "0,1,2"')
    return parser.parse_args()


def main():
    """主函数"""
    torch.cuda.empty_cache()
    gc.collect()

    args = parse_args()

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU(s): {args.gpu}")
    else:
        print("Using CPU")

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器和对应文件名
    load_data_fn, data_file = get_dataset_loader(args.dataset)
    data_path = os.path.join(args.data_path, data_file)

    # 加载数据
    data, n_classes, n_cont_features, mods, snrs, lbl, test_idx = \
        load_and_preprocess_data(load_data_fn, data_path, args.batch_size, args.data_ratio)

    model, optimizer, scheduler = define_model(n_cont_features, n_classes, args)
    print(f"Model device: {next(model.parameters()).device}")

    # 计算模型复杂度
    dummy_input = torch.randn(1, n_cont_features).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Total Parameters: {params}")
    print(f"FLOPs per sample: {flops}")

    # 训练模型
    best = train_model(
        model, optimizer, scheduler, data,
        n_epochs=args.n_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
    )

    # 最终评估
    test_score, test_throughput, test_latency = evaluate(model, data, "test", args.batch_size)
    print(f'Test score after training: {test_score:.4f}')
    print(f'Throughput after training: {test_throughput:.2f} samples/second')
    print(f'Latency after training: {test_latency * 1000:.2f} ms/sample')

    print("\nResult:")
    print(best)

    # 按SNR评估
    evaluate_by_snr(
        model=model,
        data=data,
        lbl=lbl,
        snrs=snrs,
        test_idx=test_idx,
        classes=mods,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()