import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2
import torch
import BiFusev2
import os

# ================= 配置区域 =================
# 标准室内场景最大深度设为 10.0 米 (与 BiFuse 训练一致)
# 这样 3-4米的墙壁会显示为青/绿色，而不是红色，符合你的参考图风格
DEFAULT_MAX_DEPTH = 2.0
DEPTH_COLORMAP = 'jet'
# ===========================================

def save_colored_depth(depth_data, save_path, vmin=0.0, vmax=10.0, cmap_name='jet'):
    """
    保存彩色深度图 (Jet Style)
    映射: vmin(0m)=Blue -> vmax(10m)=Red
    """
    # 1. 归一化到 0-1
    # 这里的关键是使用固定的 10m 作为分母，保证颜色含义统一
    depth_norm = (depth_data - vmin) / (vmax - vmin)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    try:
        colormap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap = cm.get_cmap(cmap_name)

    # 2. 映射颜色
    rgba = colormap(depth_norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # OpenCV 格式

    cv2.imwrite(save_path, bgr)
    print(f"  -> Saved Color Jet: {save_path}")

def save_gray_metric(depth_data, save_path, vmax=10.0):
    """
    保存标准线性灰度图 (Metric Depth)
    黑(0) -> 近, 白(255) -> 远 (>=vmax)
    """
    depth_norm = depth_data / vmax
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    gray = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(save_path, gray)
    print(f"  -> Saved Gray Metric: {save_path}")

def save_gray_inverse(depth_data, save_path, vmax=10.0):
    """
    保存逆深度灰度图 (Inverse Depth / Disparity) -> 视觉效果最好，细节最清晰
    白(255) -> 近, 黑(0) -> 远
    """
    # 加上微小值防止除零
    depth_safe = depth_data + 1e-6
    
    # 简单的反转可视化: 1 - (d/max)
    # 或者使用 1/d 进行非线性映射 (更能凸显近处细节)
    # 这里使用简单的线性反转，使得 0m=白, 10m=黑
    depth_norm = depth_data / vmax
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    inv_norm = 1.0 - depth_norm
    
    gray = (inv_norm * 255).astype(np.uint8)
    cv2.imwrite(save_path, gray)
    print(f"  -> Saved Gray Inverse (Best Visual): {save_path}")

def read_image_3d60(path, target_h=512, target_w=1024):
    # 1. 读取 BGR
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # 2. 转 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize
    if img.shape[0] != target_h or img.shape[1] != target_w:
        print(f"[Info] Resizing {img.shape[:2]} -> ({target_h}, {target_w})")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    img_vis = img.copy() # uint8 for visualization

    # 4. Normalize & Transpose
    img = img.astype(np.float32) / 255.0
    img_tensor = img.transpose(2, 0, 1) # C, H, W
    
    return img_tensor, img_vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for 3D60 Standard', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results_3d60')
    # [新增] 允许用户手动调节最大深度，默认 10.0 米
    parser.add_argument('--max_dist', type=float, default=10.0, help='Max distance for visualization (meters). Objects farther than this will be Red/White.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.img))[0]

    # --- 1. Load Model ---
    network_args = {
        'save_path': './save',
        'dnet_args': {'layers': 34, 'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]},
        'pnet_args': {'layers': 18, 'nb_tgts': 2}
    }

    print(f"Initializing model ({args.mode})...")
    if args.mode == 'supervised':
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    else:
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    
    print(f"Loading weights: {args.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(param, strict=False)
    model = model.to(device)
    model.eval()

    # --- 2. Inference ---
    print(f"Processing: {args.img}")
    img_input, img_vis = read_image_3d60(args.img)
    batch = torch.FloatTensor(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model.dnet(batch)[0]
    
    if args.mode == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    # 原始深度数据 (单位: 米)
    raw_depth = depth[0, 0, ...].cpu().numpy()

    # --- 3. Save Results ---
    print(f"Visualizing with Max Depth = {args.max_dist} meters")

    # A. 保存 Jet 彩色图 (Standard 0-10m)
    # 这次不会过曝了，因为我们没有使用 percentile 强行拉伸
    save_path_jet = os.path.join(args.output_dir, f"{img_basename}_depth_jet.png")
    save_colored_depth(raw_depth, save_path_jet, vmax=args.max_dist, cmap_name='jet')

    # B. 保存灰度图 1: Inverse (白=近，黑=远) -> 推荐看这张，细节最清晰
    save_path_inv = os.path.join(args.output_dir, f"{img_basename}_depth_gray_inverse.png")
    save_gray_inverse(raw_depth, save_path_inv, vmax=args.max_dist)

    # C. 保存灰度图 2: Metric (黑=近，白=远) -> 物理意义正确，但看起来可能偏暗
    save_path_met = os.path.join(args.output_dir, f"{img_basename}_depth_gray_metric.png")
    save_gray_metric(raw_depth, save_path_met, vmax=args.max_dist)

    # D. 组合图
    print("Generating combined plot...")
    fig = plt.figure(figsize=(10, 10))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img_vis)
    plt.title("Input RGB")
    plt.axis('off')

    plt.subplot(3, 1, 2)
    # 使用 Jet 显示，固定 0-10m 范围
    plt.imshow(raw_depth, cmap='jet', vmin=0, vmax=args.max_dist)
    plt.title(f"Predicted Depth (Jet, 0-{args.max_dist}m)")
    plt.axis('off')

    plt.subplot(3, 1, 3)
    # 显示 Inverse Gray，这是人眼看灰度图最舒服的方式
    plt.imshow(1.0 - np.clip(raw_depth/args.max_dist, 0, 1), cmap='gray')
    plt.title("Predicted Depth (Inverse Gray: White=Close)")
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{img_basename}_combined_all.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Combined plot saved to: {plot_path}")