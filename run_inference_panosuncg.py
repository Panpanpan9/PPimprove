# (bifusev2) csn@chensongnan:~/WorkSpace/Pan/BiFuse++/BiFuse++_202511$ python run_inference.py   --mode supervised   --ckpt /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_202511/pretrain/supervised_pretrain.pkl   --img /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_202511/data/22.png    --output_dir ./results_demo
'''
python run_inference_panosuncg.py \
  --mode supervised \
  --ckpt /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_Improve/BiFusev2+CRF360D/Experiments/supervised_ours_PanoSUNCG/save/model_2026-01-02-13-02-49_00132.pkl \
  --img /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_202511/data/panosuncg/pcg02_color.png \
  --output_dir Experiments/supervised_ours_PanoSUNCG/Results20251230

  python run_inference_panosuncg.py \
  --mode supervised \
  --ckpt /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_Improve/BiFusev2+CRF360D+FreDSNet/Experiments/supervised_ours_Stanford2D3D/model_2026-01-12-20-50-39_00080.pkl \
  --img /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_202511/data/stanford/std1.png \
  --output_dir /home/csn/WorkSpace/Pan/BiFuse++/BiFuse++_Improve/BiFusev2+CRF360D+FreDSNet/Experiments/supervised_ours_Stanford2D3D/Results

'''  

import numpy as np
import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm # [新增] 需要用于转换色调
import matplotlib
import cv2
import torch
import BiFusev2
import os

# 设置使用的色调，'jet' 最接近您提供的参考图效果
# 您也可以尝试 'turbo' 或 'plasma' 获得类似但更平滑的效果
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
    """
    将归一化的深度图(0-1)应用colormap并保存为彩色图片文件
    """
    # 1. 获取 matplotlib 的 colormap 对象
    # 使用 matplotlib.colormaps[name] 是新版 API，旧版可用 cm.get_cmap(name)
    try:
        colormap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        colormap = cm.get_cmap(cmap_name)

    # 2. 将灰度数据映射为 RGBA 彩色数据 (输出范围 0.0-1.0)
    colored_depth_rgba = colormap(depth_normalized)

    # 3. 转换为 RGB 并变为 0-255 的 uint8 格式
    # 取前三个通道 (RGB)，忽略 Alpha 通道
    colored_depth_rgb = (colored_depth_rgba[:, :, :3] * 255).astype(np.uint8)

    # 4. OpenCV 保存图片需要 BGR 格式，进行转换
    colored_depth_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)

    # 5. 保存
    cv2.imwrite(save_path, colored_depth_bgr)
    print(f"Successfully saved colored depth image to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for BiFuse++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'], help='Choose supervised of self-supervised model')
    parser.add_argument('--ckpt', type=str, required=True, help='Pretrain weights path (.pkl)')
    parser.add_argument('--img', type=str, required=True, help='Input panorama')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results') # [新增] 输出目录参数
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

    # 2. 读取和预处理图片
    print(f"Processing image: {args.img}")
    img_pil = imageio.imread(args.img, pilmode='RGB')
    img = img_pil.astype(np.float32) / 255.0
    
    target_h, target_w = 512, 1024
    if img.shape[0] != target_h or img.shape[1] != target_w:
        print(f"Warning: Input resolution {img.shape[:2]} mismatch! Resizing to ({target_h}, {target_w})...")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_pil = cv2.resize(img_pil, (target_w, target_h), interpolation=cv2.INTER_AREA) # 同时也resize用于显示的原始图

    [h, w, _] = img.shape
    
    # 转为 Tensor
    batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].to(device)
    
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
        # 线性归一化到 0.0 - 1.0
        normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = raw_depth # 防止除零错误

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
    plt.imshow(img_pil) # 使用 uint8 格式的原图显示更稳定
    plt.axis('off')
    plt.title("Input RGB (Resized)")

    # 下半部分：彩色深度图
    plt.subplot(2, 1, 2)
    # [修改] 将 cmap='gray' 改为指定的彩色 colormap (如 'jet')
    plt.imshow(normalized_depth, cmap=DEPTH_COLORMAP)
    plt.axis('off')
    plt.title(f"Predicted Depth (Colormap: {DEPTH_COLORMAP})")
    
    plt.tight_layout()

    # 保存组合图表 (必须在 plt.show() 之前调用)
    plot_save_path = os.path.join(args.output_dir, f"{img_basename}_combined_plot.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"Saved combined plot to: {plot_save_path}")

    # 展示结果到屏幕
    print("Showing result window...")
    plt.show()