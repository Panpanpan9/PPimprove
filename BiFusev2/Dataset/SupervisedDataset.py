import os
import sys
import cv2
import math  # 新增
import numpy as np
from imageio import imread
from tqdm import tqdm 
import io
import zipfile
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
# 新增: 引入 PyTorch3D 和 内部转换模块
import pytorch3d.transforms.rotation_conversions as p3dr
from ..Projection import EquirecRotate
from .BaseDataset import BaseDataset


# 新增: 生成随机欧拉角的辅助函数
def SampleEuler(sample_angle):
    # 生成范围在 [-sample_angle/2, sample_angle/2] 之间的随机角度
    # 例如 sample_angle=60度 (弧度制)，则范围是 [-30度, 30度]
    x = (np.random.rand() - 0.5) * sample_angle
    y = (np.random.rand() - 0.5) * sample_angle
    z = (np.random.rand() - 0.5) * sample_angle
    euler = torch.FloatTensor([[x, y, z]])
    
    return euler

def readImage(path, shape):
    # 读取 RGB 图片
    img = np.asarray(imread(path, pilmode='RGB'), np.float32) / 255.0
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: 
        img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)
    return img.transpose(2, 0, 1)

# 20251210: 自己修改的
'''
def readDepth(path, shape):
    # 修改：适配 Stanford2D3D 的深度图读取
    # 1. 使用 cv2.imread 读取原始数据 (-1 flag)
    depth = cv2.imread(path, -1)
    
    # 2. 调整大小 (使用最近邻插值保持深度值准确性)
    if depth.shape[0] != shape[0] or depth.shape[1] != shape[1]: 
        depth = cv2.resize(depth, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    
    # 3. 数值转换：uint16 -> float32 (meters)
    # Stanford2D3D 的深度值通常需要除以 512 得到米
    depth = depth.astype(np.float32) / 512.0
    
    # 4. 截断最大深度 (例如 10米)，并将无效值设为 0 (0会被 Loss 函数忽略)
    max_depth = 10.0
    depth[depth > max_depth] = 0
    depth[depth < 0] = 0
    
    return depth[None, ...] # 增加通道维度 [1, H, W]
'''
'''
# 20251210: 参考 Unifuse 对 Stanford2D3D 的对深度图的处理方式
def readDepth(path, shape):
    # 1. 读取原始 uint16 数据 (-1 标志位至关重要)
    depth = cv2.imread(path, -1)
    
    if depth is None:
        raise ValueError(f"Failed to load depth file: {path}")

    # 2. 调整大小：必须使用最近邻插值 (INTER_NEAREST) 避免边缘伪影
    if depth.shape[0] != shape[0] or depth.shape[1] != shape[1]: 
        depth = cv2.resize(depth, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    
    # 3. 数据类型转换与缩放 (参考 unifuse/stanford.py)
    depth = depth.astype(np.float32) / 512.0
    
    # 4. 清洗数据：处理 NaN 和 Inf (防止训练出现 NaN)
    # 将 NaN 和 无穷大 替换为 0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 5. 截断与掩码 (适配 BiFusev2 的 Loss)
    # BiFusev2 的 Loss 通常忽略 <= 0 的值。
    # 因此，我们将大于 max_depth 的值也设为 0，让 Loss 函数自动忽略它们。
    max_depth = 10.0  # 室内场景通常设为 10米
    depth[depth > max_depth] = 0
    depth[depth < 0] = 0  # 确保没有负值
    
    # 6. 增加通道维度 [H, W] -> [1, H, W]
    return depth[None, ...]
'''

# '''
class SupervisedDataset(BaseDataset):
    # 修改: 增加 augmentation 参数
    def __init__(self, dataset_path, mode, shape, augmentation=None, **kwargs):
        assert os.path.isdir(dataset_path) and mode in ['train', 'val']
        super().__init__(**kwargs)
        self.shape = shape
        self.augmentation = augmentation  # 保存增强配置
        
        # 新增: 初始化全景旋转模块
        if self.augmentation and 'rotate' in self.augmentation:
            self.equi_rotate = EquirecRotate(shape[0])

        # 修改：读取包含 "RGB路径 深度路径" 的 txt 文件
        # 假设 txt 文件每一行是: relative/path/to/rgb.png relative/path/to/depth.png
        self.data = []
        list_file = os.path.join(dataset_path, f'{mode}.txt')
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0: continue
                # 分割 RGB 和 深度路径
                splits = line.strip().split() 
                if len(splits) >= 2:
                    rgb_rel = splits[0]
                    depth_rel = splits[1]
                    self.data.append((
                        os.path.join(dataset_path, rgb_rel),
                        os.path.join(dataset_path, depth_rel)
                    ))
    
    def __getitem__(self, idx):
        rgb_path, depth_path = self.data[idx]

        rgb = readImage(rgb_path, self.shape)
        depth = readDepth(depth_path, self.shape)

        # 新增: 旋转增强逻辑 (移植自 SelfSupervisedDataset)
        if self.augmentation and 'rotate' in self.augmentation:
            # 获取角度范围 (例如 60度，即 +/- 30度)
            sample_angle_deg = self.augmentation['rotate']['sample_angle']
            sample_angle_rad = sample_angle_deg / 180.0 * math.pi
            
            # 1. 生成随机旋转矩阵
            euler = SampleEuler(sample_angle_rad)
            euler_R = p3dr.euler_angles_to_matrix(euler, convention='XYZ')
            
            # 2. 转换数据为 Tensor 并增加 Batch 维度 (1, C, H, W)
            rgb_tensor = torch.from_numpy(rgb).unsqueeze(0)
            depth_tensor = torch.from_numpy(depth).unsqueeze(0)
            
            # 3. 旋转 RGB (默认双线性插值)
            # 注意: 需要转置旋转矩阵以匹配 EquirecRotate 的坐标定义
            rotated_rgb = self.equi_rotate(rgb_tensor, rotation_matrix=euler_R.transpose(1, 2))
            
            # 4. 旋转 Depth (必须使用最近邻插值 mode='nearest'，防止插值产生错误深度)
            rotated_depth = self.equi_rotate(depth_tensor, rotation_matrix=euler_R.transpose(1, 2), mode='nearest')
            
            # 5. 转回 Numpy 并移除 Batch 维度
            rgb = rotated_rgb.squeeze(0).numpy()
            depth = rotated_depth.squeeze(0).numpy()

        out = {
            'idx': idx,
            'rgb': rgb,
            'depth': depth
        }

        return out
# '''

def readDepth(path, shape):
    # 参考 SelfSupervisedDataset.py 的处理方式
    # PanoSUNCG 的深度图通常存储格式特殊，这里沿用原项目的处理逻辑
    img = np.asarray(imread(path, pilmode='I'), np.float32) / 255.0
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: 
        img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    img *= 10  # 缩放系数，根据 BiFuse 源码保留
    return img[None, ...]

class SupervisedDataset_Panosuncg(BaseDataset):
    def __init__(self, dataset_path, mode, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.dataset_path = dataset_path
        
        # 解析 train.txt / val.txt
        # 你的格式是: rotated/.../color.png rotated/.../depth.png (空格分隔)
        list_file = os.path.join(dataset_path, f'{mode}.txt')
        self.data = []
        
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                # 分割 RGB 和 Depth 路径
                parts = line.split() 
                if len(parts) >= 2:
                    rgb_rel = parts[0]
                    depth_rel = parts[1]
                    self.data.append((
                        os.path.join(self.dataset_path, rgb_rel),
                        os.path.join(self.dataset_path, depth_rel)
                    ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.data[idx]

        # 读取数据
        rgb = readImage(rgb_path, self.shape)
        depth = readDepth(depth_path, self.shape)

        out = {
            'idx': idx,
            'rgb': rgb,
            'depth': depth
        }

        return out