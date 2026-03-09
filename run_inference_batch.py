'''
如何运行？
    需要指定三个参数：配置文件的路径、模型权重的路径、以及 GPU ID（默认就是 gpu 0，所以不写也可以）。

① 测试 Stanford2D3D (监督学习) 
假设你在根目录下，配置文件在 Experiments/supervised_ours_stanford/config.yaml，
                模型在 .../save/model_..._00100.pkl。
python run_inference_batch.py \
  --config Experiments/supervised_ours_Stanford2D3D/config.yaml \
  --ckpt Experiments/supervised_ours_Stanford2D3D/model_2026-01-13-03-35-08_00080.pkl\
  --gpu 0

② 测试 PanoSUNCG (自监督学习)
python run_inference_batch.py \
  --config Experiments/selfsupervised_ours_PanoSUNCG_capl/config.yaml \
  --ckpt Experiments/selfsupervised_ours_PanoSUNCG_capl/save/model_2025-xx-xx_00050.pkl \
  --gpu 0
'''

import os
import sys
import yaml
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

# 确保能找到 BiFusev2 模块
sys.path.append('../..') 
import BiFusev2
from BiFusev2 import Tools, Dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Inference for BiFuse++')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint (.pkl file)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 2. 读取配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 3. 准备输出目录 (在 val dataset 同级目录下创建 pred_result)
    # 获取 dataset_path，例如 /Media/Data/PanoSUNCG
    dataset_path = config['dataset_args']['val']['dataset_path']
    # 获取上一级目录，或者直接在 dataset_path 下创建？
    # 你的需求：val数据集同级目录下。通常 dataset_path 就是根目录。
    # 假设 dataset_path 是 ".../Stanford2D3D"，我们建立 ".../Stanford2D3D_pred_result" 或者 ".../Stanford2D3D/pred_result"
    # 这里我们采用在 dataset_path 内部或者同级建立文件夹
    save_dir = os.path.join(dataset_path, 'pred_result') 
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")
    else:
        print(f"Output directory exists: {save_dir}")

    # 4. 初始化模型
    # 根据 dataset_type 判断是监督还是自监督模型 (简单的判断逻辑)
    ds_type = config['dataset_args']['train']['dataset_type']
    if 'Supervised' in ds_type and 'Self' not in ds_type:
        print("Initializing Supervised Model...")
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**config['network_args'])
        is_supervised = True
    else:
        print("Initializing Self-Supervised Model...")
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**config['network_args'])
        is_supervised = False

    # 5. 加载权重
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 6. 实例化 Dataset (只为了获取文件列表和路径，不使用 Loader 以便获得原始文件名)
    # 使用反射机制找到对应的 Dataset 类
    DatasetClass = BiFusev2.Tools.rgetattr(BiFusev2.Dataset, config['dataset_args']['val']['dataset_type'])
    val_dataset = DatasetClass(**config['dataset_args']['val'])
    
    print(f"Total Validation Samples: {len(val_dataset)}")

    # 7. 开始推理
    # 获取当前时间戳 (格式: YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset))):
            # 获取数据 (经过了预处理的 tensor)
            data_item = val_dataset[i]
            
            # 获取原始文件路径
            # 注意：SupervisedDataset 和 SelfSupervisedDataset 的 self.data 存储结构略有不同
            # 我们需要去 Dataset 实例的 data 属性里找原始路径
            if is_supervised:
                # SupervisedDataset 的 self.data 是 [(rgb_path, depth_path), ...]
                rgb_path_orig = val_dataset.data[i][0]
            else:
                # SelfSupervisedDataset 的逻辑比较复杂，通常是 self.framelabels 
                # 这里我们假设你使用的是修改后的 Dataset，或者我们需要重新构造路径
                # 简单起见，我们尝试从 Dataset 的逻辑中获取，或者直接用 index 命名
                # 如果是 PanoSUNCG，通常是从 self.framelabels 获取
                # 这里为了稳健，如果拿不到名字，就用 index
                try:
                    # 尝试根据 PanoSUNCG 的逻辑 (这取决于你的 Dataset 具体实现)
                    # 假设 val_dataset.data 存在
                    rgb_path_orig = f"sample_{i}.png" 
                    if hasattr(val_dataset, 'data'):
                         rgb_path_orig = val_dataset.data[i][0] # 如果也是列表
                except:
                    rgb_path_orig = f"sample_{i}.png"

            # 准备 Tensor
            if is_supervised:
                # 监督学习: input 是 rgb tensor
                img_tensor = torch.from_numpy(data_item['rgb']).unsqueeze(0).to(device)
                # 推理
                pred_depth = model(img_tensor)
                # 监督模型通常返回 [pred, ...] 列表，取第一个
                if isinstance(pred_depth, (list, tuple)):
                    pred_depth = pred_depth[0]
            else:
                # 自监督学习: input 是 ref 和 tgts
                ref = torch.from_numpy(data_item['ref']).unsqueeze(0).to(device)
                # 自监督通常不需要 tgts 进行纯推理，但模型 forward 可能需要参数占位
                # 我们可以传个空的或者 dummy
                # BiFuse 自监督 forward(ref, targets)
                # 为了简单，我们只传入 ref，修改模型 forward 逻辑或者传入 dummy targets
                # 注意：BiFuse 自监督 forward 需要 ref 和 tgts
                # 我们可以构造一个伪造的 tgts 列表 (与 ref 形状相同)
                dummy_tgt = [torch.zeros_like(ref) for _ in range(config['network_args']['pnet_args']['nb_tgts'])]
                
                pred_depth, _, _ = model(ref, dummy_tgt)
                if isinstance(pred_depth, (list, tuple)):
                    pred_depth = pred_depth[0]

            # 8. 后处理与可视化
            # 将 Tensor 转回 Numpy
            pred = pred_depth.squeeze().cpu().numpy()
            
            # 反归一化 / 限制范围 (参考 run_inference.py 或 Trainer)
            # 限制在 [0, 10] 米 (室内场景常用)
            pred = np.clip(pred, 0, 10)
            
            # 归一化到 [0, 1] 以便保存为图片
            # 注意：这只是为了可视化。如果是为了科学计算，应该保存 .npy 或 .exr
            pred_vis = pred / 10.0 
            
            # 这里的可视化使用 colormap (类似 run_inference.py)
            # plt.imsave 会自动应用 colormap (默认 viridis 或 jet)
            # 或者使用 cv2 保存灰度图
            
            # 构造文件名
            filename = os.path.basename(rgb_path_orig)
            name_no_ext = os.path.splitext(filename)[0]
            save_name = f"{name_no_ext}_pred_{timestamp}.png"
            save_path = os.path.join(save_dir, save_name)

            # 保存彩色深度图 (Magma 或 Inferno 色阶通常好看)
            plt.imsave(save_path, pred_vis, cmap='jet')
            
            # 如果需要保存原始深度值 (float32)，可以取消下面这行的注释
            # np.save(save_path.replace('.png', '.npy'), pred)

    print(f"All done! Results saved to {save_dir}")

if __name__ == '__main__':
    main()