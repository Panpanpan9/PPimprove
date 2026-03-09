
# python run_inference.py --mode supervised --ckpt /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_Improve/BiFusev2+CRF360D+FreDSNet/Experiments/supervised_ours_Stanford2D3D/save/model_2026-01-09-08-01-47_00142.pkl  --img /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_202511/data/stanford/std1.png

'''
import numpy as np
import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2
import torch
import BiFusev2
import os

# 设置使用的色调
DEPTH_COLORMAP = 'jet'

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
    try:
        colormap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap = cm.get_cmap(cmap_name)

    colored_depth_rgba = colormap(depth_normalized)
    colored_depth_rgb = (colored_depth_rgba[:, :, :3] * 255).astype(np.uint8)
    colored_depth_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, colored_depth_bgr)
    print(f"Successfully saved colored depth image to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for BiFuse++')
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.img))[0]

    # 1. 初始化模型
    print(f"Initializing model in {args.mode} mode...")
    if args.mode == 'supervised':
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    elif args.mode == 'selfsupervised':
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    
    # 2. 加载权重
    print(f"Loading checkpoint: {args.ckpt}")
    if torch.cuda.is_available():
        param = torch.load(args.ckpt)
        device = torch.device("cuda")
    else:
        param = torch.load(args.ckpt, map_location='cpu')
        device = torch.device("cpu")
        
    try:
        model.load_state_dict(param, strict=True)
    except RuntimeError as e:
        print(f"WARNING: Strict loading failed. Trying strict=False...")
        model.load_state_dict(param, strict=False)

    model = model.to(device)
    model.eval()

    # 3. 读取和预处理图片
    print(f"Processing image: {args.img}")
    img_pil_original = imageio.imread(args.img, pilmode='RGB') # 保留原始数据用于显示
    
    # === [关键修改 1] 记录原始尺寸 ===
    orig_h, orig_w = img_pil_original.shape[:2]
    print(f"Original resolution: {orig_w}x{orig_h}")

    # 缩放用于网络推理 (必须是 512x1024)
    img = img_pil_original.astype(np.float32) / 255.0
    target_h, target_w = 512, 1024
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].to(device)
    
    # 4. 推理
    print("Running inference...")
    with torch.no_grad():
        outputs = model(batch)
        depth = outputs[0]
    
    if args.mode == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    # 获取低分辨率原始深度 (512x1024)
    raw_depth_lowres = depth[0, 0, ...].cpu().numpy().clip(0, 10)

    # === [关键修改 2] 先将深度图插值回原始分辨率 ===
    # 使用 INTER_CUBIC 或 INTER_LINEAR 保持平滑，避免马赛克
    print(f"Upsampling depth map back to {orig_w}x{orig_h}...")
    raw_depth = cv2.resize(raw_depth_lowres, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # === [关键修改 3] 减小去噪核的大小 ===
    # 之前的 (7,7) 太大了，会导致模糊。
    # 既然已经回到了高分辨率，用 (5,5) 或 (3,3) 比较合适，或者你可以根据效果完全去掉这一步
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    raw_depth = cv2.morphologyEx(raw_depth, cv2.MORPH_CLOSE, kernel)

    # 5. 归一化
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()
    
    if depth_max - depth_min > 1e-8:
        normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = raw_depth

    # 6. 保存高清结果
    depth_save_path = os.path.join(args.output_dir, f"{img_basename}_depth_{DEPTH_COLORMAP}_HD.png")
    save_colored_depth(normalized_depth, depth_save_path, cmap_name=DEPTH_COLORMAP)

    # 7. 生成组合图 (此时 img_pil_original 和 normalized_depth 尺寸一致，画出来更清晰)
    print("Generating output plot...")
    fig = plt.figure(figsize=(10, 8)) # 这里的尺寸仅影响显示窗口大小，不影响上面保存的HD png

    plt.subplot(2, 1, 1)
    plt.imshow(img_pil_original)
    plt.axis('off')
    plt.title(f"Input RGB ({orig_w}x{orig_h})")

    plt.subplot(2, 1, 2)
    plt.imshow(normalized_depth, cmap=DEPTH_COLORMAP)
    plt.axis('off')
    plt.title(f"Predicted Depth HD (Colormap: {DEPTH_COLORMAP})")
    
    plt.tight_layout()
    plot_save_path = os.path.join(args.output_dir, f"{img_basename}_combined_plot.png")
    # 提高 dpi 使组合图文字和细节更清晰
    plt.savefig(plot_save_path, dpi=200) 
    print(f"Saved combined plot to: {plot_save_path}")
'''

'''
import numpy as np
import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2
import torch
import BiFusev2
import os

# 设置使用的色调
DEPTH_COLORMAP = 'jet'

# 网络参数配置 (需与训练时一致)
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

    # 将灰度数据映射为 RGBA 彩色数据
    colored_depth_rgba = colormap(depth_normalized)

    # 转换为 RGB 并变为 0-255 的 uint8 格式
    colored_depth_rgb = (colored_depth_rgba[:, :, :3] * 255).astype(np.uint8)

    # OpenCV 保存图片需要 BGR 格式
    colored_depth_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, colored_depth_bgr)
    print(f"Successfully saved colored depth image to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for BiFuse++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'], help='Choose supervised of self-supervised model')
    parser.add_argument('--ckpt', type=str, required=True, help='Pretrain weights path (.pkl)')
    parser.add_argument('--img', type=str, required=True, help='Input panorama')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
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
    
    # 2. 加载权重
    print(f"Loading checkpoint: {args.ckpt}")
    if torch.cuda.is_available():
        param = torch.load(args.ckpt)
        device = torch.device("cuda")
    else:
        param = torch.load(args.ckpt, map_location='cpu')
        device = torch.device("cpu")
        
    # [关键修改] 使用 strict=True 来确保权重严格匹配
    # 如果这一步报错，说明权重没有完全加载成功
    try:
        model.load_state_dict(param, strict=True)
        print("Checkpoint loaded successfully (Strict Mode).")
    except RuntimeError as e:
        print(f"WARNING: Strict loading failed! Error: {e}")
        print("Trying with strict=False (some weights might be random)...")
        model.load_state_dict(param, strict=False)

    model = model.to(device)
    model.eval()

    # 3. 读取和预处理图片
    print(f"Processing image: {args.img}")
    img_pil = imageio.imread(args.img, pilmode='RGB')
    # 归一化到 [0, 1]
    img = img_pil.astype(np.float32) / 255.0
    
    target_h, target_w = 512, 1024
    if img.shape[0] != target_h or img.shape[1] != target_w:
        print(f"Warning: Input resolution {img.shape[:2]} mismatch! Resizing to ({target_h}, {target_w})...")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_pil = cv2.resize(img_pil, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 转为 Tensor: [1, 3, H, W]
    batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].to(device)
    
    # 4. 推理
    print("Running inference...")
    with torch.no_grad():
        # [关键修改] 直接调用 model(batch) 而不是 model.dnet(batch)
        # 这样会自动调用 model.preprocess() 进行 ImageNet 归一化
        outputs = model(batch)
        
        # BiFuse 返回的是一个列表 [depth]
        depth = outputs[0]
    
    if args.mode == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    # 获取原始深度数据 (Clip 到 0-10m 范围用于可视化)
    raw_depth = depth[0, 0, ...].cpu().numpy().clip(0, 10)

    # ================= [修改：更强力的去黑点方案] =================
    
    # 方案 A: 形态学闭运算 (Morphological Closing) - 推荐
    # 原理：先膨胀(Dilation)填满黑洞，再腐蚀(Erosion)恢复边缘。
    # kernel_size 可以尝试 (5,5) 或 (7,7)。如果黑点较大，加大这个数值。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    raw_depth = cv2.morphologyEx(raw_depth, cv2.MORPH_CLOSE, kernel)
    
    # 方案 B (备选): 如果方案 A 仍有残留，可以叠加中值滤波
    # raw_depth = cv2.medianBlur(raw_depth, 5)
    
    # =============================================================

    # 5. 归一化 (用于可视化和生成彩色图)
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()
    
    if depth_max - depth_min > 1e-8:
        # 线性归一化到 0.0 - 1.0
        normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = raw_depth

    # 6. 保存结果
    depth_save_path = os.path.join(args.output_dir, f"{img_basename}_depth_{DEPTH_COLORMAP}.png")
    save_colored_depth(normalized_depth, depth_save_path, cmap_name=DEPTH_COLORMAP)

    # 7. 生成组合图
    print("Generating output plot...")
    fig = plt.figure(figsize=(10, 8))

    # 上半部分：原图 RGB
    plt.subplot(2, 1, 1)
    plt.imshow(img_pil)
    plt.axis('off')
    plt.title("Input RGB")

    # 下半部分：彩色深度图
    plt.subplot(2, 1, 2)
    plt.imshow(normalized_depth, cmap=DEPTH_COLORMAP)
    plt.axis('off')
    plt.title(f"Predicted Depth (Colormap: {DEPTH_COLORMAP})")
    
    plt.tight_layout()

    plot_save_path = os.path.join(args.output_dir, f"{img_basename}_combined_plot.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"Saved combined plot to: {plot_save_path}")

    # (可选) 如果在无界面的服务器上运行，可以注释掉这行
    # plt.show()
'''


import numpy as np 
import argparse
from imageio import imread
import matplotlib.pyplot as plt

import torch
import BiFusev2

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for BiFuse++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    args = parser.parse_args()

    # -----------------------------------------------------
    # 1. 建立模型
    # -----------------------------------------------------
    if args.mode == 'supervised': 
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    else:
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)

    # -----------------------------------------------------
    # 2. 加载权重（只加载一次）
    # -----------------------------------------------------
    print("\n=== Loading checkpoint:", args.ckpt, "===")
    param = torch.load(args.ckpt, map_location='cpu')

    # 若 ckpt 包含 state_dict
    if isinstance(param, dict) and 'state_dict' in param:
        print(">> ckpt 中检测到 'state_dict'，自动提取")
        param = param['state_dict']

    print(">> ckpt key 数量:", len(param))

    # 执行加载
    res = model.load_state_dict(param, strict=False)
    print(">> load_state_dict 返回信息：", res)

    print(">> 模型 state_dict 中前 20 个 key：")
    print(list(model.state_dict().keys())[:20])

    model = model.cuda()
    model.eval()

    # -----------------------------------------------------
    # 3. 读入图像（保持原作者逻辑：必须是 512x1024）
    # -----------------------------------------------------
    img = imread(args.img, pilmode='RGB').astype(np.float32) / 255.0

    # 保存 raw image 供你对比
    import imageio
    # imageio.imsave('debug_in_raw.png', ( img * 255).astype('uint8'))
    imageio.imsave('debug_in_raw.png', (np.clip(img, 0.0, 1.0) * 255).astype('uint8'))

    h, w, _ = img.shape
    assert h == 512 and w == 1024, f"输入尺寸是 {h}x{w}，但 BiFuse++ 要求 512x1024"

    # -----------------------------------------------------
    # 4. 网络前向
    # -----------------------------------------------------
    batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].cuda()

    with torch.no_grad():
        raw = model.dnet(batch)[0]

    # raw 输出统计
    print("\n=== dnet raw output stats ===")
    print("min:", raw.min().item(), 
          "max:", raw.max().item(), 
          "mean:", raw.mean().item(), 
          "std:", raw.std().item())

    # -----------------------------------------------------
    # 5. 后处理（保持原作者逻辑）
    # -----------------------------------------------------
    if args.mode == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(raw) + 0.01)
    else:
        depth = raw

    depth = depth[0, 0].cpu().numpy().clip(0, 10)

    # -----------------------------------------------------
    # 6. 可视化
    # -----------------------------------------------------
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(depth, cmap='turbo')
    plt.show()
