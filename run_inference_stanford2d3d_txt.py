'''
python run_inference_stanford2d3d_txt.py \
  --mode supervised \
  --ckpt /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_Improve/BiFusev2+CRF360D+CUBE360+FreDSNet/Experiments/supervised_ours_Stanford2D3D/save/model_2026-01-13-03-35-08_00080.pkl \
  --input_txt /media/csn/81d7c547-046e-46d8-ab6f-79a6a4250b85/Stanford2D3D/val.txt \
  --output_dir /media/csn/81d7c547-046e-46d8-ab6f-79a6a4250b85/Stanford2D3D/Pred
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
from tqdm import tqdm

# ==========================================
# 引用当前项目的 BiFusev2
# ==========================================
import BiFusev2

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

# ==========================================
# 核心辅助函数
# ==========================================

def get_stanford_save_id(rel_path):
    """提取 ID：area_5b_camera_[hash]..._domain"""
    try:
        parts = rel_path.replace('\\', '/').split('/')
        area_name = parts[0]
        file_name = parts[-1]
        base_name = file_name.replace('_rgb.png', '').replace('.png', '')
        return f"{area_name}_{base_name}"
    except:
        return os.path.splitext(os.path.basename(rel_path))[0]

def process_depth_strictly(depth_map, cmap_name='jet', clip_percentile=98.0):
    """
    严格按照 convert_depth_stanford.py 逻辑执行
    """
    if depth_map is None: return None, None
    
    # 1. 转换为 float32 确保计算精度
    img_float = depth_map.astype(np.float32)

    # 2. 识别有效像素 (排除 0)
    valid_pixels = img_float[img_float > 0]
    if valid_pixels.size == 0:
        return None, None

    # 3. 计算统计范围 (严格执行 98% 截断)
    depth_min = valid_pixels.min()
    depth_max = np.percentile(valid_pixels, clip_percentile)
    
    # 4. 线性归一化到 0.0 - 1.0
    img_clipped = np.clip(img_float, depth_min, depth_max)
    if depth_max - depth_min > 1e-6:
        normalized_depth = (img_clipped - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = img_clipped * 0

    # 5. 应用 Colormap (Jet)
    try:
        colormap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap = cm.get_cmap(cmap_name)

    colored_rgba = colormap(normalized_depth)
    
    # 6. 转为 uint8 BGR 格式保存
    colored_depth_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    colored_depth_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)
    
    return colored_depth_bgr, normalized_depth

def save_combined_pure(rgb_img, pred_norm, save_path, cmap_name='jet'):
    """生成无间隙拼接图"""
    fig = plt.figure(figsize=(10, 10))
    # 上 RGB
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(rgb_img)
    ax1.axis('off')
    
    # 下 Pred (使用归一化深度图)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(pred_norm, cmap=cmap_name)
    ax2.axis('off')
    
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def preprocess_stanford(img_path):
    """imread -> /255.0 -> resize(1024, 512)"""
    img_pil = imageio.imread(img_path, pilmode='RGB')
    orig_h, orig_w = img_pil.shape[:2]
    img_norm = img_pil.astype(np.float32) / 255.0
    img_input = cv2.resize(img_norm, (1024, 512), interpolation=cv2.INTER_AREA)
    return img_input, img_pil, (orig_h, orig_w)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--input_txt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='/media/csn/81d7c547-046e-46d8-ab6f-79a6a4250b85/Stanford2D3D')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 初始化模型
    print("Loading model...")
    if args.mode == 'supervised':
        model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    else:
        model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model = model.to(device).eval()

    # 2. 读取列表
    process_list = []
    with open(args.input_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            process_list.append({'rel_rgb': parts[0], 'rel_gt': parts[1] if len(parts)>1 else None})

    # 3. 循环推理
    for item in tqdm(process_list):
        full_rgb_path = os.path.join(args.dataset_root, item['rel_rgb'])
        full_gt_path = os.path.join(args.dataset_root, item['rel_gt']) if item['rel_gt'] else None
        
        if not os.path.exists(full_rgb_path): continue
        
        # A. 预处理与模型推理
        img_in, img_orig, (oh, ow) = preprocess_stanford(full_rgb_path)
        batch = torch.FloatTensor(img_in).permute(2, 0, 1)[None, ...].to(device)
        
        with torch.no_grad():
            out = model(batch)
            pred = out[0] if isinstance(out, list) else out
            
        # B. 预测深度处理 (LANCZOS4 锐化 + 双边滤波)
        pred_np = pred[0, 0, ...].cpu().numpy()
        pred_orig_size = cv2.resize(pred_np, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        pred_refined = cv2.bilateralFilter(pred_orig_size, d=7, sigmaColor=0.1, sigmaSpace=5)
        
        # 应用严格的 98% 截断逻辑
        pred_colored, pred_norm = process_depth_strictly(pred_refined, clip_percentile=98.0)

        # C. 命名与保存
        save_id = get_stanford_save_id(item['rel_rgb'])
        shutil.copy(full_rgb_path, os.path.join(args.output_dir, f"{save_id}_color.png"))
        
        # D. GT 处理 (严格使用 cv2.IMREAD_UNCHANGED 读取 16-bit)
        if full_gt_path and os.path.exists(full_gt_path):
            # 关键修复点：使用 UNCHANGED 保持 16bit 连续性
            gt_img_raw = cv2.imread(full_gt_path, cv2.IMREAD_UNCHANGED)
            if gt_img_raw is not None:
                gt_colored, _ = process_depth_strictly(gt_img_raw, clip_percentile=98.0)
                cv2.imwrite(os.path.join(args.output_dir, f"{save_id}_gt.png"), gt_colored)
        
        # E. 保存预测图与拼接图
        cv2.imwrite(os.path.join(args.output_dir, f"{save_id}_pred_cc80.png"), pred_colored)
        # save_combined_pure(img_orig, pred_norm, os.path.join(args.output_dir, f"{save_id}_comb_cc80.png"))

if __name__ == '__main__':
    main()