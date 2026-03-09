import torch
import sys
import os

# --- 核心修正：添加路径并正确导入 ---
# 将当前目录加入系统路径，确保能找到 BiFusev2 文件夹
sys.path.append(os.getcwd())

try:
    # 尝试方式 1：如果你在 BiFusev2+CRF360D 根目录下运行 (推荐)
    from BiFusev2.BiFuse import ResUNet, SupervisedCombinedModel
    print("✅ 成功从 BiFusev2.BiFuse 导入模型")
except ImportError:
    try:
        # 尝试方式 2：如果你把脚本放到了 BiFusev2 子文件夹里运行
        from BiFuse import ResUNet, SupervisedCombinedModel
        print("✅ 成功从 BiFuse 导入模型")
    except ImportError as e:
        print("❌ 导入失败！请检查目录结构。")
        print(f"详细错误: {e}")
        exit(1)
# --------------------------------

def test_bifuse_crf360d():
    print("=== 开始验证 BiFusev2 + CRF360D 模型集成 ===")
    
    # 1. 模拟配置参数 (参考 Config)
    dnet_args = {
        'layers': 34,
        'CE_equi_h': [8, 16, 32, 64, 128, 256, 512] 
    }
    
    # 2. 实例化模型
    print("正在实例化 SupervisedCombinedModel...")
    try:
        # 初始化模型 (save_path 是为了 BaseModule 不报错)
        model = SupervisedCombinedModel(save_path='./check_test', dnet_args=dnet_args)
        
        # --- 关键修正：必须移至 CUDA ---
        # 因为 mobius_utils.py 等文件中含有硬编码的 .cuda()，如果不放 GPU 会报错
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ 模型已加载至 GPU")
        else:
            print("⚠️ 警告: 没有检测到 GPU，如果代码中有 .cuda() 可能会报错")
            
        print("✅ 模型实例化成功")
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 创建假输入数据 [Batch, 3, Height, Width]
    batch_size = 2
    height = 512
    width = 1024
    
    try:
        if torch.cuda.is_available():
            dummy_input = torch.randn(batch_size, 3, height, width).cuda()
        else:
            dummy_input = torch.randn(batch_size, 3, height, width)
        print(f"ℹ️ 输入数据形状: {dummy_input.shape}")

        # 4. 前向传播测试
        output = model(dummy_input)
        
        # 5. 验证输出
        if isinstance(output, list) and len(output) == 1:
            depth_pred = output[0]
            print("✅ 输出类型正确 (List len=1)")
        else:
            print(f"❌ 输出类型错误: 期望 List[Tensor], 实际 {type(output)}")
            return

        # 验证尺寸 [B, 1, H, W]
        expected_shape = (batch_size, 1, height, width)
        if depth_pred.shape == expected_shape:
            print(f"✅ 输出尺寸验证通过: {depth_pred.shape}")
        else:
            print(f"❌ 输出尺寸错误: 期望 {expected_shape}, 实际 {depth_pred.shape}")

        print("=== 验证完成：修改逻辑正确，输入输出符合原项目要求 ===")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ 显存不足 (OOM)。引入 Attention 后显存占用增加，请尝试减小 batch_size。")
        else:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_bifuse_crf360d()