'''
python run_inference_panosuncg_txt.py   --mode supervised   --config Experiments/supervised_ours_PanoSUNCG/config.yaml   --input_txt Experiments/supervised_ours_PanoSUNCG/val.txt   --ckpt Experiments/supervised_ours_PanoSUNCG/model_2026-01-12-12-13-50_00107.pkl
'''

import numpy as np
import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2
import torch
import os
import shutil
import yaml
from tqdm import tqdm
import BiFusev2

DEPTH_COLORMAP = 'jet' 

network_args = {
    'save_path': './save',
    'dnet_args': {'layers': 34, 'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]},
    'pnet_args': {'layers': 18, 'nb_tgts': 2}
}

# --- 辅助函数 ---
def parse_dataset_root_from_yaml(config_path):
    """从 config.yaml 提取数据集路径"""
    dataset_root = ""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
                dataset_root = config_dict.get('dataset_path', '')
                if not dataset_root:
                    try: dataset_root = config_dict['dataset_args']['val']['dataset_path']
                    except: pass
                if dataset_root:
                    print(f"Loaded dataset_path from config: {dataset_root}")
            except:
                print("Warning: Could not parse config.yaml.")
    return dataset_root

def parse_stanford_filename(rel_path):
    try:
        rel_path = rel_path.replace('\\', '/')
        parts = rel_path.split('/')
        area_id = parts[0]
        filename = parts[-1]
        file_parts = filename.split('_')
        # 兼容处理 camera_hash_room_idx 这种格式
        cam_hash = file_parts[1][:8]
        room_name = file_parts[2]
        return f"{area_id}-{room_name}-{cam_hash}"
    except:
        return os.path.splitext(os.path.basename(rel_path))[0]

def get_colored_depth(depth_numpy, cmap_name='jet', vmin=0.0, vmax=10.0):
    depth_clipped = np.clip(depth_numpy, vmin, vmax)
    depth_norm = (depth_clipped - vmin) / (vmax - vmin)
    try: colormap = matplotlib.colormaps[cmap_name]
    except: colormap = cm.get_cmap(cmap_name)
    colored_rgba = colormap(depth_norm)
    colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)

def save_combined_plot_matplotlib(rgb_img, pred_depth, save_path, cmap_name='jet', vmin=0.0, vmax=10.0):
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1); plt.imshow(rgb_img); plt.axis('off'); plt.title("Input RGB")
    plt.subplot(2, 1, 2); plt.imshow(pred_depth, cmap=cmap_name, vmin=vmin, vmax=vmax); plt.axis('off')
    plt.title(f"Predicted Depth (Range: {vmin}-{vmax}m)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)

def preprocess_image(img_path, target_h=512, target_w=1024):
    img_pil = imageio.imread(img_path, pilmode='RGB')
    orig_h, orig_w = img_pil.shape[:2]
    img = img_pil.astype(np.float32) / 255.0
    img_input = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img_input, img_pil, (orig_h, orig_w)

def main():
    parser = argparse.ArgumentParser(description='Batch Inference for Stanford2D3D')
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--input_txt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()

    # 1. 确定路径逻辑 (参考 Panosuncg)
    dataset_root = ""
    if args.config:
        dataset_root = parse_dataset_root_from_yaml(args.config)
    
    # 如果 config 里没写，或者手动指定了，在这里回退
    if not dataset_root:
        dataset_root = "/media/csn/81d7c547-046e-46d8-ab6f-79a6a4250b85/Stanford2D3D"

    save_root = args.output_dir if args.output_dir else os.path.join(dataset_root, 'Pred')
    os.makedirs(save_root, exist_ok=True)
    print(f"Dataset root: {dataset_root}\nResults will be saved to: {save_root}")

    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'supervised':
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    else:
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model = model.to(device).eval()

    # 3. 循环处理
    with open(args.input_txt, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in tqdm(lines):
        splits = line.split()
        rel_rgb = splits[0]
        rel_depth = splits[1] if len(splits) > 1 else None
        
        # --- 核心 Hotfix: 路径纠错逻辑 ---
        # 针对 Stanford2D3D 可能存在的 'rgb_/' 到 'pano/rgb/' 的转换
        rgb_path = os.path.join(dataset_root, rel_rgb)
        if not os.path.exists(rgb_path):
            # 尝试插入 pano 目录
            alt_rgb = os.path.join(dataset_root, rel_rgb.replace('area_', 'area_').replace('rgb_/', 'pano/rgb/'))
            if os.path.exists(alt_rgb):
                rgb_path = alt_rgb
            else:
                continue # 如果还是找不到，跳过

        base_id = parse_stanford_filename(rel_rgb)

        # A. 预处理与推理
        img_input, img_orig, orig_size = preprocess_image(rgb_path)
        batch = torch.FloatTensor(img_input).permute(2, 0, 1)[None, ...].to(device)

        with torch.no_grad():
            depth_out = model(batch)
            depth = depth_out[0] if isinstance(depth_out, list) else depth_out
        
        if args.mode == 'selfsupervised':
            depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

        pred_depth_small = depth[0, 0, ...].cpu().numpy()
        pred_depth_orig = cv2.resize(pred_depth_small, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_CUBIC)

        # B. 保存
        # ① RGB
        shutil.copy(rgb_path, os.path.join(save_root, f"{base_id}_color.png"))
        # ② GT (同样应用 Hotfix)
        if rel_depth:
            gt_path = os.path.join(dataset_root, rel_depth)
            if not os.path.exists(gt_path):
                gt_path = os.path.join(dataset_root, rel_depth.replace('depth/', 'pano/depth/'))
            
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
                if gt_img is not None:
                    gt_color = get_colored_depth(gt_img.astype(np.float32), vmin=0.0, vmax=10.0)
                    cv2.imwrite(os.path.join(save_root, f"{base_id}_gt.png"), gt_color)

        # ③ Pred
        pred_color = get_colored_depth(pred_depth_orig, vmin=0.0, vmax=10.0)
        cv2.imwrite(os.path.join(save_root, f"{base_id}_pred_cc80.png"), pred_color)
        # ④ Comb
        save_combined_plot_matplotlib(img_orig, pred_depth_orig, os.path.join(save_root, f"{base_id}_comb.png"))

if __name__ == '__main__':
    main()