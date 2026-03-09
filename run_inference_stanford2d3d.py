'''
python run_inference_stanford2d3d.py  \
   --mode supervised   --ckpt Experiments/supervised_ours_Stanford2D3D/save/model_2026-01-13-03-35-08_00080.pkl  \
   --img data/stanford/std/area_5a_camera_ddcf23b53da948e296418014ee672498_hallway_15_frame_equirectangular_domain_color.png  \
   --output_dir Experiments/supervised_ours_Stanford2D3D/Results_20260112

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
DEPTH_COLORMAP = 'jet' # 或 'plasma', 'magma'

# 模型参数配置 (必须与训练时完全一致)
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

def save_colored_depth(depth_numpy, save_path, cmap_name='jet', vmin=0.0, vmax=10.0):
    """
    将深度图保存为彩色图片
    depth_numpy: 原始深度值 (meters)
    vmin, vmax: 可视化的深度范围 (米)
    """
    # 1. 截断范围
    depth_clipped = np.clip(depth_numpy, vmin, vmax)
    
    # 2. 归一化到 0-1
    # 注意：不要使用 (d - min) / (max - min)，因为单张图的 min/max 不稳定
    # 使用固定的 vmin/vmax 能保证不同图片颜色含义一致
    depth_normalized = (depth_clipped - vmin) / (vmax - vmin)

    # 3. 获取 colormap
    try:
        colormap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap = cm.get_cmap(cmap_name)

    # 4. 映射颜色 (RGBA)
    colored_depth_rgba = colormap(depth_normalized)

    # 5. 转为 RGB uint8
    colored_depth_rgb = (colored_depth_rgba[:, :, :3] * 255).astype(np.uint8)

    # 6. 转 BGR 供 OpenCV 保存
    colored_depth_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, colored_depth_bgr)
    print(f"Successfully saved colored depth image to: {save_path}")

def preprocess_image(img_path, target_h=512, target_w=1024):
    """
    读取并预处理图像
    """
    # 1. 读取 RGB 图片
    img_pil = imageio.imread(img_path, pilmode='RGB')
    orig_h, orig_w = img_pil.shape[:2]
    
    # 2. 归一化到 [0, 1] float32
    img = img_pil.astype(np.float32) / 255.0
    
    # 3. 尺寸调整 (强制 Resize 到模型训练尺寸 512x1024)
    # 注意：BiFusev2+CRF360D 内部可能有位置编码与分辨率强绑定，
    # 所以输入必须是 512x1024，不能直接传 2048x4096
    if img.shape[0] != target_h or img.shape[1] != target_w:
        print(f"Resize input: {img.shape[:2]} -> ({target_h}, {target_w}) for inference")
        img_input = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        img_input = img
        
    return img_input, img_pil, (orig_h, orig_w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for Stanford2D3D', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'], help='Choose supervised or self-supervised model')
    parser.add_argument('--ckpt', type=str, required=True, help='Pretrain weights path (.pkl)')
    parser.add_argument('--img', type=str, required=True, help='Input panorama path')
    parser.add_argument('--output_dir', type=str, default='./results_stanford2d3d', help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.img))[0]

    # ==========================================
    # 1. 初始化模型
    # ==========================================
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
        
    # 【关键修改】打印未加载的键，确保新模块 (SAM, PanoSA) 权重被加载
    load_res = model.load_state_dict(param, strict=False)
    if len(load_res.missing_keys) > 0:
        print("WARNING: Missing keys in checkpoint (Using random init for these):")
        # 只打印前几个，避免刷屏
        for k in load_res.missing_keys[:5]: print(f"  - {k}")
        if len(load_res.missing_keys) > 5: print(f"  ... and {len(load_res.missing_keys)-5} more.")
    else:
        print("Success: All keys matched!")

    model = model.to(device)
    model.eval()

    # ==========================================
    # 2. 读取和预处理
    # ==========================================
    print(f"Processing image: {args.img}")
    # img_input: 512x1024 float32 (给模型)
    # img_orig: 原始尺寸 uint8 (给显示)
    img_input, img_orig, orig_size = preprocess_image(args.img)
    
    batch = torch.FloatTensor(img_input).permute(2, 0, 1)[None, ...].to(device)

    # ==========================================
    # 3. 推理 (Inference)
    # ==========================================
    print("Running inference...")
    with torch.no_grad():
        # 【修正】直接调用 model(batch)，它会自动调用内部的 self.preprocess(batch) 进行标准化
        # SupervisedCombinedModel 的 forward 返回的是列表 [depth]
        depth_list = model(batch)
        depth = depth_list[0]
    
    if args.mode == 'selfsupervised':
        depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    # 获取原始深度数据 (512x1024)
    pred_depth_small = depth[0, 0, ...].cpu().numpy()

    # ==========================================
    # 4. 后处理与可视化优化
    # ==========================================
    
    # 【优化 1】将预测的深度图上采样回原始图像尺寸 (例如 2048x4096)
    # 使用双三次插值 (CUBIC) 或 线性插值 (LINEAR) 可以使大图看起来更平滑，减少马赛克感
    # 注意：深度图是浮点数，resize 是安全的
    print(f"Upsampling depth map to original size: {orig_size}")
    pred_depth_orig = cv2.resize(pred_depth_small, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_CUBIC)

    # 【优化 2】基于固定范围归一化 (Stanford2D3D 室内深度通常在 0-10米)
    # 这样对比度更符合物理意义
    save_colored_depth(pred_depth_orig, 
                       os.path.join(args.output_dir, f"{img_basename}_depth_{DEPTH_COLORMAP}.png"), 
                       cmap_name=DEPTH_COLORMAP, 
                       vmin=0.0, vmax=10.0) # 室内一般最远10米

    # [保存组合图]
    print("Generating combined plot...")
    fig = plt.figure(figsize=(15, 10)) # 增大画布

    # 原图
    plt.subplot(2, 1, 1)
    plt.imshow(img_orig) 
    plt.axis('off')
    plt.title(f"Input RGB {orig_size}")

    # 深度图 (使用相同的 colormap 和 vmin/vmax)
    plt.subplot(2, 1, 2)
    plt.imshow(pred_depth_orig, cmap=DEPTH_COLORMAP, vmin=0.0, vmax=10.0)
    plt.axis('off')
    plt.title(f"Predicted Depth (Resized back to Original)")
    
    plt.tight_layout()
    plot_save_path = os.path.join(args.output_dir, f"{img_basename}_combined_plot.png")
    # plt.savefig(plot_save_path, dpi=150)
    # print(f"Saved combined plot to: {plot_save_path}")
