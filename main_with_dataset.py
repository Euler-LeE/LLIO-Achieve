# 这个文件里我尝试进行了数据集的加载，但是没有成功。
# 我提供了一个简单的示例，这是可以运行的训练流程。
# 目前的大多数数据集都不太适配，但是我仍然保留了此文件。为后续的尝试做准备。
# This file contains my attempt to load a dataset, which was unsuccessful. 
# I have provided a simple example that demonstrates a working training process.
# Most of the datasets currently available are not well-suited for this purpose, 
# but I have kept this file for future reference and experimentation.

import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_twolayer import TwoLayerModel

def download_and_extract_dataset(url, extract_to):
    dataset_zip = os.path.join(extract_to, 'dataset.zip')
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    print(f"从 {url} 下载数据集...")
    urllib.request.urlretrieve(url, dataset_zip)
    
    print("解压数据集...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(dataset_zip)
    print("数据集已准备好，位于", extract_to)

def load_dataset(data_dir):
    """从指定目录加载数据集并返回训练和测试的DataLoader。"""
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def main():
    # 定义数据集的URL和本地路径
    dataset_url = "https://"
    dataset_dir = "./data"

    download_and_extract_dataset(dataset_url, dataset_dir)

    train_loader, test_loader = load_dataset(dataset_dir)

    model_para = {
        "input_len": 100,
        "input_channel": 6,
        "patch_len": 25,
        "feature_dim": 512,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": {
            "name": "ResMLP",
            "layer_num": 6,
            "expansion": 2,
            "dropout": 0.2,
        },
        "reg": {
            "name": "MeanMLP",
            "layer_num": 3,
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoLayerModel(model_para).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, 训练损失: {train_loss:.4f}, 测试损失: {test_loss:.4f}")

    print("训练完成。")

if __name__ == "__main__":
    main()
