from math import exp 
import os
import sys

sys.path.append(os.path.dirname(__file__))

import einops
import torch
from torch import nn

from einops.layers.torch import *
from torch.nn.modules.dropout import Dropout

from model_MLP import *  # 保持原有的MLP模块
class SimpleMLPReg(nn.Module):
    def __init__(self, patch_num=10, feature_len=10, out_dim=3, layer_num=4, active_fun=nn.GELU(), dropout=0.5):
        '''
        INPUT: [Batch_size Patch_num Feature_lenght]
        OUTPUT: [Batch_size, out_dim] [Batch_size, out_dim] (y, y_cov)
        '''

        super(SimpleMLPReg, self).__init__()
        self.input_len = int(patch_num * feature_len)
        self.dropout = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            Rearrange('b l f -> b (l f)'),
            *[nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                self.dropout
            )
                for i in range(layer_num)
            ]
            # nn.Linear(self.input_len, out_dim * 2)
        )
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim)

    def forward(self, x):
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)

class PoolingMLPReg(nn.Module):
    def __init__(self, patch_num=10,
                 feature_len=10,
                 out_dim=3,
                 layer_num=4,
                 active_fun=nn.GELU(),
                 dropout=0.5,
                 pooling_type='mean'):
        '''
        INPUT: [Batch_size Patch_num Feature_lenght]
        OUTPUT: [Batch_size, out_dim] [Batch_size, out_dim] (y, y_cov)
        '''

        super(PoolingMLPReg, self).__init__()
        self.input_len = int(feature_len)
        self.dropout = nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            Reduce('b l f-> b f', pooling_type),
            *[nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                active_fun,
                self.dropout
            )
                for i in range(layer_num)
            ]
        )
        self.out_linear = nn.Linear(self.input_len, out_dim)
        self.out2_linear = nn.Linear(self.input_len, out_dim)

    def forward(self, x):
        out = self.net(x)
        return self.out_linear(out), self.out2_linear(out)


# 新增 VO 特征提取模块（视觉里程计）
class VOFeatureExtractor(nn.Module):
    def __init__(self, input_channel=3, output_dim=128):
        super(VOFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(128 * 8 * 8, output_dim)  # 我假设输入图片尺寸为 32x32，还是得看具体数据

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.fc(features)

# ResMLP 提取器 (没动，保持原有结构)
class ResMLPExtractor(nn.Module):
    def __init__(self, patch_num=10, patch_len=10, input_channel=6, mlp_in_dim=15, expansion=4, active_func=nn.GELU(), layer_num=4, dropout=0.5):
        super(ResMLPExtractor, self).__init__()

        def wrapper(i, fn): return PreAffinePostLayerScale(mlp_in_dim, i, fn)

        self.net = nn.Sequential(
            Rearrange('b c (l w) -> b l (w c)', w=patch_len),
            nn.Linear(int(patch_len * input_channel), mlp_in_dim),
            *[
                nn.Sequential(
                    wrapper(i, nn.Conv1d(patch_num, patch_num, 1, bias=False)),
                    wrapper(i, nn.Sequential(
                        nn.Linear(mlp_in_dim, mlp_in_dim * expansion, bias=False),
                        active_func,
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(mlp_in_dim * expansion, mlp_in_dim, bias=False)
                    ))
                ) for i in range(layer_num)
            ],
            Affine(mlp_in_dim)
        )

    def forward(self, x):
        return self.net(x)


# 合并 VO 特征和 LLIO 特征的模型
class TwoLayerFusionModel(nn.Module):
    def __init__(self, model_para=None):
        super(TwoLayerFusionModel, self).__init__()

        if model_para is None:
            model_para = {
                "input_len": 100,
                "input_channel": 6,
                "patch_len": 10,
                "feature_dim": 20,
                "out_dim": 3,
                "active_func": "GELU",
                "extractor": {
                    "name": "ResMLP",
                    "layer_num": 4,
                    "expansion": 4,
                },
                "reg": {
                    "name": "MLP",
                    "layer_num": 3,
                },
                "vo": {
                    "input_channel": 3,
                    "output_dim": 128
                }
            }

        # 激活函数的设置
        self.active_function = nn.GELU() if model_para["active_func"] == "GELU" else nn.ReLU()
        patch_num = int(model_para["input_len"] / model_para["patch_len"])

        # 提取 IMU 特征
        if model_para["extractor"]["name"] == "ResMLP":
            self.extractor = ResMLPExtractor(
                patch_num=patch_num,
                patch_len=model_para["patch_len"],
                input_channel=model_para["input_channel"],
                mlp_in_dim=model_para["feature_dim"],
                expansion=model_para["extractor"]["expansion"],
                active_func=self.active_function,
                layer_num=model_para["extractor"]["layer_num"],
                dropout=model_para["extractor"].get("dropout", 0.5)
            )

        # 提取 VO 特征
        self.vo_extractor = VOFeatureExtractor(
            input_channel=model_para["vo"]["input_channel"],
            output_dim=model_para["vo"]["output_dim"]
        )

        # 回归层，接收融合后的特征
        self.reg = SimpleMLPReg(
            patch_num=patch_num,
            feature_len=model_para["feature_dim"] + model_para["vo"]["output_dim"],
            layer_num=model_para["reg"]["layer_num"],
            active_fun=self.active_function
        )

    def forward(self, imu_data, image_data):
        imu_features = self.extractor(imu_data) 
        vo_features = self.vo_extractor(image_data) 

        # 简单的融合了特征
        combined_features = torch.cat((imu_features, vo_features.unsqueeze(1).expand(-1, imu_features.size(1), -1)), dim=2)
        
        out1, out2 = self.reg(combined_features)
        return out1, out2


if __name__ == '__main__':
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
        },
        "vo": {
            "input_channel": 3,
            "output_dim": 128
        }
    }

    net = TwoLayerFusionModel(model_para)
    imu_data = torch.rand([512, 6, 100])  # IMU 数据: batch_size, input_channel, input_len
    image_data = torch.rand([512, 3, 32, 32])  # 图像数据: batch_size, channels, height, width

    y, y_cov = net(imu_data, image_data)
    print('IMU 输入:', imu_data.shape, '图像输入:', image_data.shape)
    print('输出 y:', y.shape, '输出 y_cov:', y_cov.shape)
