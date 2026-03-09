'''
python run_inference_stanford2d3d.py \
  --mode supervised \
  --ckpt Experiments/supervised_ours_3D60/save/model_2026-01-22-04-18-50_00091.pkl \
  --img /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_202511/data/3D60/01_color.png \
  --output_dir Experiments/supervised_ours_3D60/Results20260121_jet

'''
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2  # [重点] 3D60 数据集使用 cv2 处理图像
import torch
import BiFusev2
import os

# 设置使用的色调
DEPTH_COLORMAP = 'jet'

# 模型参数配置
network_args = {
    'save_path': './save',
    'dnet_args': {
        'layers': 34,
        'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]
    },
    'pnet_args': {
        'layers': 18,
        'nb_tgts': 2
    }
}

def save_colored_depth(depth_normalized, save_path, cmap_name='jet'):
    """
    将归一化的深度图(0-1)应用colormap并保存为彩色图片文件
    """
    try:
        colormap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap = cm.get_cmap(cmap_name)

    # 映射为 RGBA
    colored_depth_rgba = colormap(depth_normalized)

    # 转为 RGB (0-255)
    colored_depth_rgb = (colored_depth_rgba[:, :, :3] * 255).astype(np.uint8)

    # 转为 BGR (OpenCV 保存格式)
    colored_depth_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, colored_depth_bgr)
    print(f"Successfully saved colored depth image to: {save_path}")

def read_image_3d60(path, target_h=512, target_w=1024):
    """
    参考 SupervisedDataset_3D60.py 中的 readImage 函数进行处理
    """
    # 1. 使用 OpenCV 读取 (BGR)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # 2. 转为 RGB (Dataset 逻辑)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. 调整大小 (使用 INTER_AREA)
    if img.shape[0] != target_h or img.shape[1] != target_w:
        print(f"Resizing input from {img.shape[:2]} to ({target_h}, {target_w})...")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 准备用于 Matplotlib 显示的 uint8 图片
    img_vis = img.copy()

    # 4. 归一化并转置: [H, W, C] -> [C, H, W] (Dataset 逻辑)
    img = img.astype(np.float32) / 255.0
    img_tensor_input = img.transpose(2, 0, 1)
    
    return img_tensor_input, img_vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for BiFuse++ on 3D60 Dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'], help='Choose supervised of self-supervised model')
    parser.add_argument('--ckpt', type=str, required=True, help='Pretrain weights path (.pkl)')
    parser.add_argument('--img', type=str, required=True, help='Input panorama path')
    parser.add_argument('--output_dir', type=str, default='./results_3d60', help='Directory to save results')
    args = parser.parse_args()

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.img))[0]

    # 1. 初始化模型
    print(f"Initializing model in {args.mode} mode...")
    if args.mode == 'supervised':
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    elif args.mode == 'selfsupervised':
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    
    print(f"Loading checkpoint: {args.ckpt}")
    if torch.cuda.is_available():
        param = torch.load(args.ckpt)
        device = torch.device("cuda")
    else:
        param = torch.load(args.ckpt, map_location='cpu')
        device = torch.device("cpu")
        
    model.load_state_dict(param, strict=False)
    model = model.to(device)
    model.eval()

    # 2. 读取和预处理图片 (3D60 专用逻辑)
    print(f"Processing image: {args.img}")
    try:
        # img_input: [3, 512, 1024], float32, 0-1
        # img_vis: [512, 1024, 3], uint8, RGB
        img_input, img_vis = read_image_3d60(args.img)
    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)

    # 转为 Tensor: 增加 batch 维度 [1, 3, 512, 1024]
    batch = torch.FloatTensor(img_input).unsqueeze(0).to(device)
    
    # 3. 推理
    print("Running inference...")
    with torch.no_grad():
        depth = model.dnet(batch)[0]
    
    if args.mode == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    # 获取原始深度数据 (Clip 到 0-10m 范围用于可视化)
    raw_depth = depth[0, 0, ...].cpu().numpy().clip(0, 10)

    # 4. 归一化 (用于可视化和生成彩色图)
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()
    
    if depth_max - depth_min > 1e-8:
        normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = raw_depth 

    # ------------------------------------------------------------------
    # [保存] 保存单独的彩色深度图文件
    # ------------------------------------------------------------------
    depth_save_path = os.path.join(args.output_dir, f"{img_basename}_depth_{DEPTH_COLORMAP}.png")
    save_colored_depth(normalized_depth, depth_save_path, cmap_name=DEPTH_COLORMAP)

    # ------------------------------------------------------------------
    # [展示与保存组合图] 使用 Matplotlib 绘图并展示
    # ------------------------------------------------------------------
    print("Generating output plot...")
    fig = plt.figure(figsize=(10, 8))

    # 上半部分：原图 RGB
    plt.subplot(2, 1, 1)
    # img_vis 已经是 RGB 格式的 uint8 数据，可以直接显示
    plt.imshow(img_vis) 
    plt.axis('off')
    plt.title("Input RGB (Resized, 3D60 Preprocess)")

    # 下半部分：彩色深度图
    plt.subplot(2, 1, 2)
    plt.imshow(normalized_depth, cmap=DEPTH_COLORMAP)
    plt.axis('off')
    plt.title(f"Predicted Depth (Colormap: {DEPTH_COLORMAP})")
    
    plt.tight_layout()

    # 保存组合图表
    plot_save_path = os.path.join(args.output_dir, f"{img_basename}_combined_plot.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"Saved combined plot to: {plot_save_path}")

    # 展示结果
    print("Showing result window...")
    plt.show()