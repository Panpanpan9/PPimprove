# 20260105 新增 configure_optimizers 类（不同阶段设置不用的学习率）
'''
# 20260108 修改 
    新增 mobius_utils.py：这是 CRF360D 中处理球面坐标变换的基础工具。
    新增 PanoLayers.py：这是 CRF360D 的核心网络层，包含 PanoSA（用于 Encoder 修正）和 PanoLayer（用于 Decoder 增强）。
    重构 BiFuse.py：
        FusionModule: 改造为串行结构。先通过 PanoSA 修正 ERP 特征，再通过 Cross-Attention (CUBE360) 与 Cube 特征融合。
        ResUNet: 在解码器中插入 PanoLayer (SF-CRFs)。
    <注意> 跳跃连接环节暂时保持原 bifusev2 的(使用cat)，不使用 SAM 
'''

import sys
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     # 25060105 新增 引入 optim
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.models as models
import functools
import math
from .BaseModule import BaseModule
from .CETransform import CETransform

# 20251228
from .PanoLayers import PanoSA, PanoLayer
# from timm.models.layers import to_2tuple # 如果需要

# 20260104  跳跃连接环节 用 SAM 代替 原来的concat
from .SAM import SAM    # 引用 PixelFormer 的 SAM 组件

# 202601xx 【暂时还未调整：将骨干网络升级为 EfficientNet-B5】
from .panocrf import EfficientNetEncoder

# 权重初始化工具:初始化网络参数，而且要为不同层（Conv / DeConv / BN）设置不同初始化方式
def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d): 
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# ResNet：封装预训练的 ResNet 模型，用于提取图像的​​多层次特征​​（浅层纹理到深层语义）【特征提取】
class ResNet(nn.Module):
    # 只要代码中 pretrained=True（默认就是 True），程序启动时就会自动加载 ImageNet 权重。
    def __init__(self, layers, in_channels=3, pretrained=True):
        assert layers in [18, 34, 50, 101, 152]
        super().__init__()
        # 加载预训练的 ResNet 模型（如 ResNet-34）: 自动从网络下载在 ImageNet 上训练好的权重
        # pretrained_model = models.__dict__['resnet{}'.format(layers)](weights='ResNet%d_Weights.DEFAULT'%layers)
        # 【20251228 完善】注意：PyTorch新版本中建议使用 weights 参数，旧版本使用 pretrained=True
        try:
            pretrained_model = models.__dict__['resnet{}'.format(layers)](weights='ResNet%d_Weights.DEFAULT'%layers)
        except:
            pretrained_model = models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            # 输入为 3 通道（RGB），直接使用预训练的 conv1 和 bn1
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            # 输入通道数非 3（如等距投影图），自定义 conv1 和 bn1
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)    # 应用自定义权重初始化
            weights_init(self.bn1)
        
        # 共享 ResNet 的后续层（relu、maxpool、layer1-layer4）
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']   # 浅层特征（低层次纹理）
        self.layer2 = pretrained_model._modules['layer2']   # 中层特征（中等层次纹理区域）
        self.layer3 = pretrained_model._modules['layer3']   # 深层特征（高层次语义）
        self.layer4 = pretrained_model._modules['layer4']   # 最深层特征（全局上下文）

        # clear memory 释放预训练模型内存（避免冗余）
        del pretrained_model

    def forward(self, x):
        # resnet 前向传播
        x = self.conv1(x)   # 初始卷积（7x7, stride=2, padding=3）
        x = self.bn1(x)     # 批归一化
        x0 = self.relu(x)   # ReLU 激活
        x = self.maxpool(x0)    # 最大池化（3x3, stride=2）
        x1 = self.layer1(x)     # 浅层特征（layer1 输出）
        x2 = self.layer2(x1)    # 中层特征（layer2 输出）
        x3 = self.layer3(x2)    # 深层特征（layer3 输出）
        x4 = self.layer4(x3)    # 最深层特征（layer4 输出）

        # 返回各层特征（用于后续融合）
        out = {
            'l0': x0,   # 初始卷积后特征
            'l1': x1,   # layer1 输出
            'l2': x2,   # layer2 输出
            'l3': x3,   # layer3 输出
            'l4': x4    # layer4 输出
        }

        return out
    
    # 预处理
    def preforward(self, x):
        # 简化版前向传播（仅到 maxpool 层）
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        return x

class SinusoidalPositionEmbedder(nn.Module):
    """
    CUBE360 Positional Encoding
    """
    def __init__(self, multires):
        super().__init__()
        self.multires = multires
        self.frequencies = 2.0 ** torch.linspace(0.0, self.multires - 1, self.multires)

    def forward(self, coords):
        # coords: [B, H, W, 2]
        embed = [coords]
        for freq in self.frequencies:
            embed.append(torch.sin(coords * freq * math.pi))
            embed.append(torch.cos(coords * freq * math.pi))
        return torch.cat(embed, dim=-1)
    

# FusionModule：融合等距投影图（equi）和立方体展开图（cube）的特征，解决两种视角的特征对齐问题 【特征融合】
class FusionModule(nn.Module):
    """
    Serial Fusion: 
    1. CRF Correction (PanoSA) -> Fixes ERP Distortion
    2. CUBE Fusion (Cross-Attention) -> Blends Global & Local Context
    """
    # --- 20251228 改进点 2: 引入 PSI (PanoSA) 【开始】---
    # 20251228 修改点 1: 初始化时接收 height 和 width
    # 【重要修复】这里必须加上 num_heads=4
    def __init__(self, num_channels, CE, num_heads=4):
    # def __init__(self, num_channels, CE, height, width):
        super().__init__()
        self.CE = CE    # 坐标转换器

        self.num_channels = num_channels
        
        # --- 1. CRF Correction Module ---
        self.psi = PanoSA(
            dim=num_channels,
            num_heads=4,             # 可以根据显存调整，通常 4 或 8
            window_size=4,           # 窗口大小
            # shift_size=0,            # 融合阶段暂不使用 shift 以简化 mask 处理
            rotation=True,           # 开启 SWT (球面窗口变换)
            localconv=True,
            interact=True
        )
        # --- 20251228 改进点 2: 引入 PSI (PanoSA) 【结束】

        # --- 2. CUBE Fusion Module (Position Encoding) ---
        self.pos_embedder = SinusoidalPositionEmbedder(multires=6)
        pe_dim = 2 + 2 * 2 * 6 
        self.pe_proj = nn.Linear(pe_dim, num_channels)

        # --- 3. Cross Attention Modules ---
        # Enhance Cube using Corrected Equi context
        self.attn_cube = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        # Enhance Equi using Cube context
        self.attn_equi = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)

        self.norm_equi = nn.LayerNorm(num_channels)
        self.norm_cube = nn.LayerNorm(num_channels)
        
        self.ff_equi = nn.Sequential(nn.Linear(num_channels, num_channels), nn.ReLU(), nn.Linear(num_channels, num_channels))
        self.ff_cube = nn.Sequential(nn.Linear(num_channels, num_channels), nn.ReLU(), nn.Linear(num_channels, num_channels))

        # Final Fusion Conv
        self.conv_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        
        # Cube Convolution  【20251228 新增】conv_cube：用于处理 Cube 投影特征的卷积
        self.conv_cube = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.conv_equi_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv_cube_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    # 20260108 新增 【开始】
    def _get_equi_coords(self, N, H, W, device):
        y_rng = torch.linspace(-1, 1, H, device=device)
        x_rng = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_rng, x_rng, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)

    def _get_cube_coords(self, N, H_face, W_face, device):
        try:
            keys = list(self.CE.e2c.keys())
            key = keys[-1] 
            grids = []
            for i in range(6):
                g = getattr(self.CE.e2c[key], f'grid_{i}')
                grids.append(g)
            full_grid = torch.cat(grids, dim=0)
            full_grid = full_grid.permute(0, 3, 1, 2)
            full_grid = F.interpolate(full_grid, size=(H_face, W_face), mode='bilinear', align_corners=True)
            full_grid = full_grid.permute(0, 2, 3, 1)
            return full_grid.unsqueeze(0).repeat(N, 1, 1, 1, 1)
        except:
            return torch.zeros(N, 6, H_face, W_face, 2, device=device)
    # 20260108 新增 【结束】

    def forward(self, equi, cube):
        # cube 输入形状: [B, C, 6, h, w] (来自 ResUNet 的 reshape)
        B, C, _, h_c, w_c = cube.shape
        
        # -----------------------------------------------------------
        # Fix 1: Reshape Cube for C2E (5D -> 4D)
        # -----------------------------------------------------------
        # 将 [B, C, 6, h, w] 转换为 [B*6, C, h, w] 以适应 CETransform
        cube_stacked = cube.transpose(1, 2).reshape(B * 6, C, h_c, w_c)
        
        # 1. Cube Projection & Convolution
        f_cube_proj = self.CE.C2E(cube_stacked)     # Project Cube back to Equi [B, C, H_e, W_e]
        f_cube_proj = self.conv_cube(f_cube_proj) 

        # -----------------------------------------------------------
        # Step 1: CRF Correction (PanoSA) on Equi Features
        # -----------------------------------------------------------
        B, C, H_e, W_e = equi.shape
        self.psi.H = H_e
        self.psi.W = W_e
        
        flat_equi = equi.flatten(2).transpose(1, 2) # [B, HW, C]

        # Apply PanoSA correction
        flat_equi_corrected = self.psi(flat_equi, mask_matrix=None)
        # flat_equi_corrected: 经过 PanoSA（球面窗口自注意力）修正后的全景图特征，形状被展平（Flatten）以输入 Transformer。

        # -----------------------------------------------------------
        # Step 2: CUBE Fusion (Cross-Attention)
        # -----------------------------------------------------------
        # Prepare Cube Features (Positional Embeddings)
        device = equi.device
        coords_equi = self._get_equi_coords(B, H_e, W_e, device)
        coords_cube = self._get_cube_coords(B, h_c, w_c, device)
        # pe_equi / pe_cube: Positional Embedding（位置编码），用于告诉 Transformer 像素点在球面或立方体面上的几何位置。
        pe_equi = self.pe_proj(self.pos_embedder(coords_equi.reshape(B, -1, 2)))
        pe_cube = self.pe_proj(self.pos_embedder(coords_cube.reshape(B, -1, 2)))

        # cube is [B, C, 6, h, w], flatten(2) merges last 3 dims -> [B, C, 6*h*w]
        flat_cube = cube.flatten(2).transpose(1, 2) # [B, 6hw, C]

        # Key/Value Pooling for Efficiency
        # cube_kv 这里的 cube 需要用原始的 5D 张量或者 stacked 4D 张量来处理
        # 这里用 cube_stacked [B*6, C, h, w] 做池化比较方便
        cube_kv = F.adaptive_avg_pool2d(cube_stacked, (8, 8)) # [B*6, C, 8, 8]
        cube_kv = cube_kv.view(B, 6, C, 8, 8).flatten(3).transpose(2, 3).reshape(B, -1, C)
        
        pool_h, pool_w = 16, 32
        equi_kv_feat = F.adaptive_avg_pool2d(equi, (pool_h, pool_w))
        equi_kv = equi_kv_feat.flatten(2).transpose(1, 2)

        # A. Enhance Cube (Query=Cube, Key=Corrected Equi)
        q_cube = flat_cube + pe_cube
        # Let's pool the CORRECTED feature for better context
        equi_corrected_map = flat_equi_corrected.transpose(1, 2).reshape(B, C, H_e, W_e)
        equi_kv_corrected = F.adaptive_avg_pool2d(equi_corrected_map, (pool_h, pool_w)).flatten(2).transpose(1, 2)
        
        out_cube, _ = self.attn_cube(query=q_cube, key=equi_kv_corrected, value=equi_kv_corrected)
        flat_cube = self.norm_cube(flat_cube + out_cube)
        flat_cube = flat_cube + self.ff_cube(flat_cube)

        # B. Enhance Equi (Query=Corrected Equi, Key=Cube)
        q_equi = flat_equi_corrected + pe_equi # Use Corrected Equi as Query base
        k_cube = cube_kv
        
        out_equi, _ = self.attn_equi(query=q_equi, key=k_cube, value=k_cube)
        flat_equi_final = self.norm_equi(flat_equi_corrected + out_equi)
        flat_equi_final = flat_equi_final + self.ff_equi(flat_equi_final)

        # -----------------------------------------------------------
        # Step 3: Final Reshape and Fusion
        # -----------------------------------------------------------
        f_equi_final = flat_equi_final.transpose(1, 2).reshape(B, C, H_e, W_e)
        # f_cube_final (Transformer Output) is not directly projected back, 
        # typically we use the projection path for skip connections.
        
        # Fusion for Skip Connection
        f_cat = torch.cat([f_equi_final, f_cube_proj], dim=1) 
        
        f_equi_res = equi + self.conv_equi_cat(f_cat)
        f_cube_res = f_cube_proj + self.conv_cube_cat(f_cat) # [B, C, H_e, W_e]
        f_fusion = self.conv_cat(f_cat)

        # -----------------------------------------------------------
        # Fix 2: Reshape Output Cube for ResUNet (4D -> 5D)
        # -----------------------------------------------------------
        # E2C 返回 [B*6, C, h, w]
        f_cube_res_stacked = self.CE.E2C(f_cube_res) 
        # ResUNet 期望收到 [B, C, 6, h, w]，因为它后面紧接着做了 transpose(1, 2)
        # [B*6, C, h, w] -> [B, 6, C, h, w] -> [B, C, 6, h, w]
        f_cube_res_out = f_cube_res_stacked.view(B, 6, C, h_c, w_c).transpose(1, 2)

        return f_equi_res, f_cube_res_out, f_fusion


# 深度估计网络 ResUNet: 基于 ResNet 和特征融合模块（FusionModule）的深度估计网络，输入 RGB 图像（或等距/立方体投影图），输出场景的深度图
class ResUNet(nn.Module):
    def __init__(self, layers, CE_equi_h):
        super().__init__()
        # 定义通道数（ResNet-34 输出 512 通道，ResNet-50+ 输出 2048 通道）
        num_channels = 512 if layers <= 34 else 2048
        # 1. 定义两个 ResNet 骨干网络（分别处理等距和立方体输入）
        self.resnet_equi = ResNet(layers, in_channels=3, pretrained=True)   # resnet_equi 专门处理 Equirectangular
        self.resnet_cube = ResNet(layers, in_channels=3, pretrained=True)   # resnet_cube 专门处理 Cubemap
        
        # 定义 坐标转换工具：用于在 Equi 和 Cube 之间的转换，这在后面融合时非常关键
        self.ce = CETransform(CE_equi_h)

        # 定义四个融合模块（对应 ResNet 的 layer1-layer4）
        self.f1 = FusionModule(num_channels//8, self.ce)    # 浅层特征融合  （layer1）
        self.f2 = FusionModule(num_channels//4, self.ce)    # 中层特征融合  （layer2）
        self.f3 = FusionModule(num_channels//2, self.ce)    # 深层特征融合  （layer3）
        self.f4 = FusionModule(num_channels//1, self.ce)    # 最深层特征融合（layer4）

        # 定义反卷积上采样模块（恢复分辨率）
        planes = [num_channels, num_channels//2, num_channels//4, num_channels//8]
        # ResNet34：plaes[0] 512   plaes[1] 256     plaes[2] 128     plaes[3] 64
        
        def create_conv(in_ch, out_ch, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size-1)//2),  # 卷积
                nn.BatchNorm2d(out_ch), # 批归一化
                nn.ReLU()   # ReLU 激活
            )
        
        
        # 定义解码层 即反卷积层（结合跳跃连接）
        # =================== 【20260104 修改】 调整 Deconv 通道数 =====================
        # ------------ Decoder Block 4  ------------
        self.deconv4 = nn.Sequential(
            # Deconv4 不需要改，因为它是解码器的起点，fusion_4 直接输入，没有 Skip Connection 拼接
            create_conv(planes[0], planes[1]*4, kernel_size=3), # 卷积扩展通道
            nn.PixelShuffle(2)  # 像素洗牌上采样（×2 分辨率）：上采样 2倍，通道数变为 1/4
        )
        self.sf_crf4 = PanoLayer(embed_dim=planes[1], window_size=4, num_heads=4, rotation=True)

        # ------------ Decoder Block 3 (SAM) ------------
        # 【20260104 新增 】Layer 3 的 SAM
        # self.sam3 = SAM(input_dim=planes[1], embed_dim=planes[1], v_dim=planes[1], window_size=7, num_heads=4)
        
        self.deconv3 = nn.Sequential(
            create_conv(planes[0], planes[2]*4, kernel_size=3),
            # create_conv(planes[1], planes[2]*4, kernel_size=3), # sam 修改输入通道 planes[0] -> planes[1]
            nn.PixelShuffle(2)
        )
        self.sf_crf3 = PanoLayer(embed_dim=planes[2], window_size=4, num_heads=4, rotation=True)

        # ------------ Decoder Block 2 (SAM) ------------
        # 【20260104 新增 】Layer 2 的 SAM
        # self.sam2 = SAM(input_dim=planes[2], embed_dim=planes[2], v_dim=planes[2], window_size=7, num_heads=4)
        
        self.deconv2 = nn.Sequential(
            create_conv(planes[1], planes[3]*4, kernel_size=3),
            # create_conv(planes[2], planes[3]*4, kernel_size=3), # sam 修改输入通道 planes[1] -> planes[2]
            nn.PixelShuffle(2)
        )
        self.sf_crf2 = PanoLayer(embed_dim=planes[3], window_size=4, num_heads=4, rotation=True)

        # ------------ Decoder Block 1 (SAM) ------------
        # 【20260104 新增 】Layer 1 的 SAM
        # self.sam1 = SAM(input_dim=planes[3], embed_dim=planes[3], v_dim=planes[3], window_size=7, num_heads=4)

        self.deconv1 = nn.Sequential(
            create_conv(planes[2], planes[3]//2*4, kernel_size=3),
            # create_conv(planes[3], planes[3]//2*4, kernel_size=3), # sam 修改输入通道 planes[2] -> planes[3]
            nn.PixelShuffle(2),     # 上采样
            nn.Conv2d(planes[3]//2, 1, kernel_size=1),  # 最终卷积输出单通道深度图
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # 上采样到原图尺寸：最后一次双线性插值调整到原图尺寸
        )
        # --- 修改点 3: 在解码器加入 SF-CRFs (PanoLayer) 【结束】---

    def forward(self, x):   # x 是输入的 equi全景图
        # 2. 预处理 (Pre-forward)
        # 步骤1：提取等距和立方体输入的 ResNet 特征
        f_equi = self.resnet_equi.preforward(x)     # equi 输入的 ResNet 特征（l0-l4）
        # Cube 分支 吃 转换成 Cube 格式的 x (self.ce.E2C(x))
        f_cube = self.resnet_cube.preforward(self.ce.E2C(x))    # cube 输入先转换为等距特征，再提取 ResNet 特征
        real_B = x.shape[0] # 20260108 新增

        # --- Encoder & Fusion ---
        # 步骤2：逐层融合特征（layer1-layer4）
        # Layer 1 处理
        f_equi_l1 = self.resnet_equi.layer1(f_equi)     # equi 分支跑 Layer1 输入的 layer1 特征
        f_cube_l1 = self.resnet_cube.layer1(f_cube)     # cube 分支跑 Layer1 输入的 layer1 特征
        # 融合！更新后的特征传入下一层，fusion_1 留给解码器
        # f_equi_l1, f_cube_l1, fusion_1 = self.f1(f_equi_l1, f_cube_l1)  # 融合 layer1 特征
        _, C1, h1, w1 = f_cube_l1.shape
        f_cube_l1_reshaped = f_cube_l1.view(real_B, 6, C1, h1, w1).transpose(1, 2)
        f_equi_l1, f_cube_l1_reshaped, fusion_1 = self.f1(f_equi_l1, f_cube_l1_reshaped)
        f_cube_l1 = f_cube_l1_reshaped.transpose(1, 2).reshape(real_B*6, C1, h1, w1)

        # Layer 2 处理
        f_equi_l2 = self.resnet_equi.layer2(f_equi_l1)  # equi 输入的 layer2 特征
        f_cube_l2 = self.resnet_cube.layer2(f_cube_l1)  # cube 输入的 layer2 特征
        # f_equi_l2, f_cube_l2, fusion_2 = self.f2(f_equi_l2, f_cube_l2)  # 融合 layer2 特征
        _, C2, h2, w2 = f_cube_l2.shape
        f_cube_l2_reshaped = f_cube_l2.view(real_B, 6, C2, h2, w2).transpose(1, 2)
        f_equi_l2, f_cube_l2_reshaped, fusion_2 = self.f2(f_equi_l2, f_cube_l2_reshaped)
        f_cube_l2 = f_cube_l2_reshaped.transpose(1, 2).reshape(real_B*6, C2, h2, w2)

        # Layer 3 处理
        f_equi_l3 = self.resnet_equi.layer3(f_equi_l2)  # equi 输入的 layer3 特征
        f_cube_l3 = self.resnet_cube.layer3(f_cube_l2)  # cube 输入的 layer3 特征
        # f_equi_l3, f_cube_l3, fusion_3 = self.f3(f_equi_l3, f_cube_l3)  # 融合 layer3 特征
        _, C3, h3, w3 = f_cube_l3.shape
        f_cube_l3_reshaped = f_cube_l3.view(real_B, 6, C3, h3, w3).transpose(1, 2)
        f_equi_l3, f_cube_l3_reshaped, fusion_3 = self.f3(f_equi_l3, f_cube_l3_reshaped)
        f_cube_l3 = f_cube_l3_reshaped.transpose(1, 2).reshape(real_B*6, C3, h3, w3)

        # Layer 4 处理
        f_equi_l4 = self.resnet_equi.layer4(f_equi_l3)  # equi 输入的 layer4 特征
        f_cube_l4 = self.resnet_cube.layer4(f_cube_l3)  # cube 输入的 layer4 特征
        # f_equi_l4, f_cube_l4, fusion_4 = self.f4(f_equi_l4, f_cube_l4)  # 融合 layer4 特征
        _, C4, h4, w4 = f_cube_l4.shape
        f_cube_l4_reshaped = f_cube_l4.view(real_B, 6, C4, h4, w4).transpose(1, 2)
        f_equi_l4, f_cube_l4_reshaped, fusion_4 = self.f4(f_equi_l4, f_cube_l4_reshaped)
        f_cube_l4 = f_cube_l4_reshaped.transpose(1, 2).reshape(real_B*6, C4, h4, w4)


        # =========【20251228 补充】Decoder & SF-CRFs =========
        # =========【20260104 补充】Decoder & SAM & SF-CRFs =========
        # 步骤3：通过反卷积上采样恢复分辨率（结合跳跃连接）
        # 1. 从最深层开始   Block 4
        feat = self.deconv4(fusion_4)   # 融合最深层特征（layer4）并上采样×2
        feat = self.sf_crf4(feat)   # 20251228 新增 Apply SF-CRF

        # 2. 跳跃连接：拼接上一层的融合特征 fusion_3    Block 3     Layer 4 -> Layer 3
        # feat = self.sam3(e=fusion_3, q=feat)    # 20260104 修改 使用 SAM 替代 Concat
        feat = torch.cat([feat, fusion_3], dim=1)   # 原作者 拼接 layer3 融合特征（跳跃连接）
        feat = self.deconv3(feat)   # 上采样×2
        feat = self.sf_crf3(feat)   # 20251228 新增 Apply SF-CRF

        # 3. 继续 跳跃连接 fusion_2     Block 2     Layer 3 -> Layer 2
        # feat = self.sam2(e=fusion_2, q=feat)
        feat = torch.cat([feat, fusion_2], dim=1)   # 原作者 拼接 layer2 融合特征
        feat = self.deconv2(feat)   # 上采样×2
        feat = self.sf_crf2(feat)   # 20251228 新增 Apply SF-CRF
        
        # 4. 最后一步       Block 1     Layer 2 -> Layer 1
        # feat = self.sam1(e=fusion_1, q=feat)
        feat = torch.cat([feat, fusion_1], dim=1)    # 原作者 拼接 layer1 融合特征
        depth = self.deconv1(feat)  # 上采样×2 并输出单通道深度图
        # 最后一步的特殊处理 (deconv1) ： deconv1 不仅要上采样，还要输出单通道的深度图 （ReUnet init 定义过 deconv1）


        return [depth]  # 返回深度图列表（兼容多输出场景）



# 姿态估计网络 PoseNet: 基于 ResNet 的姿态估计网络，输入参考图像和目标图像，输出目标的姿态参数（如旋转矩阵、平移向量）和关键点热图。
class PoseNet(nn.Module):
    def __init__(self, layers, nb_tgts):
        super().__init__()
        self.nb_tgts = nb_tgts  # 目标数量（如关键点数量）
        num_channels = 512 if layers <= 34 else 2048    # 通道数与 ResNet 层数相关
        # 初始化 ResNet（输入为参考图像 + 目标图像的拼接）
        self.resnet = ResNet(layers, 3*(nb_tgts+1), pretrained=True)
        # 姿态预测头（输出 6*nb_tgts 维向量，每目标 6 个姿态参数）
        self.pose_pred = nn.Sequential(
            nn.Conv2d(num_channels, num_channels//2, kernel_size=1),    # 1x1 卷积降维
            nn.ReLU(),  # ReLU 激活
            nn.BatchNorm2d(num_channels//2),    # 批归一化
            nn.Conv2d(num_channels//2, 6*nb_tgts, kernel_size=1)    # 输出姿态参数
        )
        # 扩展特征生成模块（用于关键点热图）
        self.exp_layers = nn.ModuleList([])
        for i in range(4):
            c = num_channels // (2**i)  # 通道数逐层减半
            l = nn.Sequential(
                nn.Conv2d(c, nb_tgts, kernel_size=3, stride=1, padding=1,bias=False),   # 3x3 卷积
                nn.Sigmoid(),   # Sigmoid 激活（生成概率热图）
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)    # 上采样×4
            )
            self.exp_layers.append(l)
        self.exp_layers = self.exp_layers[::-1]     # 反转顺序（从深层到浅层）

        # 初始化权重
        self.pose_pred.apply(weights_init)
        self.exp_layers.apply(weights_init)
    
    def forward(self, ref, tgts: list):
        # 步骤1：拼接参考图像和目标图像（作为 ResNet 输入）
        x = torch.cat([ref] + tgts, dim=1)  # 输入形状：[B, 3*(nb_tgts+1), H, W]
        res_out = self.resnet(x)    # ResNet 前向传播，获取各层特征

        # 步骤2：预测姿态参数（6*nb_tgts 维向量）
        pose = self.pose_pred(res_out['l4']).mean(-1).mean(-1)  # 对特征图全局平均池化
        pose = 0.01 * pose.view(-1, self.nb_tgts, 6)    # 调整尺度并重塑形状

        # 步骤3：生成扩展特征（关键点热图）
        exp_lst = []
        for i, (key, val) in enumerate(res_out.items()):
            if key == 'l0': continue    # 跳过初始卷积层特征
            exp_lst.append(self.exp_layers[i-1](val))   # 通过扩展层生成热图
        
        return exp_lst, pose    # 返回扩展特征和姿态参数


# 监督学习模型 SupervisedCombinedModel: 监督学习场景下的端到端深度估计模型，封装了深度网络（ResUNet）和预处理逻辑（标准化）
    # ​​输入​​：原始 RGB 图像（形状 [B, 3, H, W]）。
    # ​​输出​​：预测的 深度图（形状 [B, 1, H, W]）。
class SupervisedCombinedModel(BaseModule):
    # 设置 图像数据标准化（Normalization）参数​​
    # ImageNet 数据集的统计均值和标准差，几乎所有基于 ImageNet 预训练的 CNN 模型（如代码中的 ResNet）在训练时都会使用这一标准化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, save_path, dnet_args, pnet_args=None):
        # 调用 BaseModule 构造（用于模型文件管理）
        super().__init__(save_path)
        # 深度网络（ResUNet）
        self.dnet = ResUNet(**dnet_args)
    '''
    # ------------------ 【20260105 新增】 策略A：配置优化器与调度器 ------------------
    def configure_optimizers(self):
        # 1. 优化器 (Strategy A: 使用较小的学习率 1e-4)
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
        # 2. 学习率调度器 (Strategy A: MultiStepLR at 80, 120)
        # Gamma=0.1 表示在里程碑处将学习率乘以 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
        
        return [optimizer], [scheduler]
    # ---------------------------------------------------------------------
    '''

    # ================= 【20260105 新增】 策略B：分层学习率核心实现 =================
    def configure_optimizers(self):
        # 1. 识别 Backbone (ResNet) 的参数 ID
        # 这些参数已经预训练过，我们需要以较小的学习率(1e-4)进行微调，防止破坏预训练特征
        backbone_ids = list(map(id, self.dnet.resnet_equi.parameters())) + \
                       list(map(id, self.dnet.resnet_cube.parameters()))
        
        # 2. 分离参数组
        backbone_params = []
        new_module_params = []
        
        for name, param in self.named_parameters():
            if id(param) in backbone_ids:
                backbone_params.append(param)
            else:
                # 包括 FusionModule(PSI), SAM, Decoder(Deconv+SF-CRFs) 等随机初始化的模块
                # 这些需要更大的学习率(3e-4)来快速收敛
                new_module_params.append(param)
        
        # 3. 配置分层优化器
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-4},      # Backbone: 微调（CRF360D）
            {'params': new_module_params, 'lr': 3e-4}     # New Modules: 快速学习
        ], betas=(0.9, 0.999))
        
        # 4. 配置学习率调度器 (Scheduler)
        # 在 Epoch 80 和 120 时，将所有组的学习率乘以 0.1
        # 这能帮助模型在训练后期跳出局部最优，进一步降低 Loss
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
        
        print(f"✅ Optimizer Configured with Strategy B: Backbone(1e-4) & NewModules(3e-4)")
        return [optimizer], [scheduler]
    # ==============================================================

    def _normalize(self, img):
        # 图像标准化（匹配 ImageNet 分布）
        tmp = img.clone()
        tmp[:, 0, ...] -= self.MEAN[0]  # R 通道减均值
        tmp[:, 1, ...] -= self.MEAN[1]  # G 通道减均值
        tmp[:, 2, ...] -= self.MEAN[2]  # B 通道减均值
        tmp[:, 0, ...] /= self.STD[0]   # R 通道除以标准差
        tmp[:, 1, ...] /= self.STD[1]   # G 通道除以标准差
        tmp[:, 2, ...] /= self.STD[2]   # B 通道除以标准差
        
        return tmp

    def preprocess(self, img):
        # 预处理输入图像（标准化）
        img = self._normalize(img)

        return img

    def forward(self, batch):
        # 前向传播（监督学习模式）
        batch = self.preprocess(batch)  # 标准化输入图像
        depth = self.dnet(batch)        # 通过 ResUNet 预测深度图

        # 20250910 添加 打印 depth 形状 方便调试
        # print("-------- Bifuse.py中的SupervisedCombinedModel类中的forward函数中: --------")
        # print("pred_depth[0].shape:", depth[0].shape)   # pred_depth[0].shape: torch.Size([8, 1, 512, 1024])
        # print("gt_depth.shape:", depth[0].shape)        # gt_depth.shape: torch.Size([8, 1, 512, 1024])
        # 两者在 batch、通道、高度、宽度 上完全匹配。
        # 之前报错的尺寸不匹配 (tensor a (3) must match tensor b (1024)) 并不是因为 pred_depth 和 gt_depth 的 shape 不一致

        return depth    # 返回深度图列表

# ------------------------------------------------------- 自监督 <= DepthNet + PoseNet -------------------------------------------------------
# 自监督学习模型 SelfSupervisedCombinedModel： 自监督学习场景下的模型，同时输出深度图、扩展特征和姿态参数（无需显式标签）
    # ​​输入​​：参考图像（ref） 和 目标图像列表（tgts）。
    # 输出​​：参考图像的深度图（ref_depth）、扩展特征（exp_lst）、姿态参数（pose）。
class SelfSupervisedCombinedModel(BaseModule):
     # 归一化常数（ImageNet）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, save_path, dnet_args, pnet_args):
         # 调用 BaseModule 构造（用于模型文件管理）
        super().__init__(save_path)
        # 深度网络（ResUNet）
        self.dnet = ResUNet(**dnet_args)
        # 位姿网络（PoseNet）
        self.pnet = PoseNet(**pnet_args)
    
    def _normalize(self, img):
        # 对输入图像做 channel-wise 归一化（in-place 做到 tmp） ???
        tmp = img.clone()
        tmp[:, 0, ...] -= self.MEAN[0]
        tmp[:, 1, ...] -= self.MEAN[1]
        tmp[:, 2, ...] -= self.MEAN[2]
        tmp[:, 0, ...] /= self.STD[0]
        tmp[:, 1, ...] /= self.STD[1]
        tmp[:, 2, ...] /= self.STD[2]
        
        return tmp

    def preprocess(self, img):
        # 封装归一化接口（方便后续扩展）
        img = self._normalize(img)

        return img

    def forward(self, ref, tgts):
        # ref：参考帧 [B,3,H,W]
        # tgts：目标帧列表
        ref = self.preprocess(ref)                  # 归一化 ref
        tgts = [self.preprocess(x) for x in tgts]   # 逐个归一化目标帧
        
        # dnet 输出 inverse-depth（或 depth 相关表示）；在代码中 dnet 返回的是一个 list（多尺度）???
        ref_inv_depth = self.dnet(ref)
        # 将网络输出通过 sigmoid 缩放到 [0,1] 后按论文/实现的缩放因子变为 inverse depth（alpha=10, +0.01 偏移）
        ref_inv_depth = [10 * torch.sigmoid(x) + 0.01 for x in ref_inv_depth]
        # 将 inverse depth 转为 depth（1 / inv）
        ref_depth = [1 / x for x in ref_inv_depth]
        
        # pnet 输出 explainability masks 与 pose
        exp_lst, pose = self.pnet(ref, tgts)

        # 返回：ref_depth (list，多尺度)、explainability（这里只取第一个尺度 exp_lst[0:1]）以及 pose
        return ref_depth, exp_lst[0:1], pose