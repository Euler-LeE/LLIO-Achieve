# 轻量级学习惯性里程计
论文代码: "LLIO: 轻量级学习惯性里程计"

## 预备条件

使用 pip 安装依赖:
```bash
pip install torch einops numpy
```

## 使用方法

### 模型结构

LLIO 模型现在支持融合视觉里程计（VO）特征。`model_twolayer.py` 文件中定义了新的 `TwoLayerFusionModel` 类，该类集成了 IMU 特征提取和 VO 特征提取，并将两者融合后进行回归。

### 初始化模型

```python
model_para = {
    "input_len": 100,
    "input_channel": 6,
    "patch_len": 25,
    "feature_dim": 512,
    "out_dim": 3,
    "active_func": "GELU",
    "extractor": {  # 包括：特征转换与 ResMLP 模块，如图 3 所示。
        "name": "ResMLP",
        "layer_num": 6,
        "expansion": 2,
        "dropout": 0.2,
    },
    "reg": {  # 回归部分，如图 3 所示
        "name": "MeanMLP",
        "layer_num": 3,
    },
    "vo": {  # 视觉里程计特征提取
        "input_channel": 3,
        "output_dim": 128
    }
}

net = TwoLayerFusionModel(model_para)  # 初始化模型
```

### 前向传播

```python
imu_data = torch.rand([512, 6, 100])  # IMU 数据: batch_size, input_channel, input_len
image_data = torch.rand([512, 3, 32, 32])  # 图像数据: batch_size, channels, height, width

y, y_cov = net(imu_data, image_data)  # 输出: [batch_size, 3], [batch_size, 3]

print('IMU 输入:', imu_data.shape, '图像输入:', image_data.shape)
print('输出 y:', y.shape, '输出 y_cov:', y_cov.shape)
```

示例输出:
```bash
IMU 输入: torch.Size([512, 6, 100]) 图像输入: torch.Size([512, 3, 32, 32])
输出 y: torch.Size([512, 3]) 输出 y_cov: torch.Size([512, 3])
```

## 致谢
感谢 TLIO [https://github.com/CathIAS/TLIO]。

## 许可证
源代码根据 GPLv3 许可证发布。
