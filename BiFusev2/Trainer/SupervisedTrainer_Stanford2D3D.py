'''
20260108    目前的训练代码确实没有使用频域损失
20260112    开始使用频域损失
'''

import os
import torch
from .BaseTrainer import BaseLitModule
from pytorch_lightning import loggers as pl_loggers
from .. import Loss
from .. import Tools 

class SupervisedLitModule_Stanford2D3D(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.berHu = Loss.ReverseHuberLoss()
        # 【20260112 新增】初始化频域损失
        self.fft_loss = Loss.FrequencyDomainLoss()

    #     # 20251214【新增】定义 ImageNet 的均值和标准差用于反归一化
    #     # 定义 ImageNet 的均值和标准差用于反归一化
    #     # 使用 register_buffer 确保它们会自动随模型移动到 GPU
    #     self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    #     self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    # # 20251214【新增】反归一化
    # def denormalize(self, img_tensor):
    #     """
    #     反归一化：将标准化后的 Tensor 还原为 [0, 1] 的可视化图像
    #     img_tensor: [3, H, W]
    #     注意：img_tensor 必须与 imagenet_mean/std 在同一个设备上 (通常是 GPU)
    #     """
    #     # 反向操作：image = (image * std) + mean
    #     return img_tensor.mul(self.imagenet_std).add(self.imagenet_mean).clamp(0, 1)


    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']
        gt_depth = batch['depth']
        pred_depth = self.model(rgb)
        # 1. 原有 BerHu Loss
        berhu_loss = self.berHu(pred_depth[0], gt_depth)

        # 2. 【20260212 新增】计算频域损失
        # 权重 lambda 可以调整，通常设为 0.1 或 0.05，避免主导 BerHu Loss
        lambda_fft = 0.1 
        fft_loss_val = self.fft_loss(pred_depth[0], gt_depth)
        
        # 3. 【20260112 新增】总损失
        total_loss = berhu_loss + lambda_fft * fft_loss_val

        out = {
            'loss': total_loss,
            'berhu_loss': berhu_loss.detach(), # 记录分项 Loss 方便观察
            'fft_loss': fft_loss_val.detach()
        }
        # out = {
        #     'loss': berhu_loss,
        # }

        # 20251210 添加了旋转噪声，所以修改 write_logger，同时还需修改 training_step 来调用这个更新后的 write_logger
        # self.write_logger(out)    # 原作者代码
        self.write_logger(out, rgb=rgb, pred=pred_depth[0], gt=gt_depth)    # 调用 write_logger 并传入用于可视化的数据
        
        return out
    

    def validation_step(self, batch, batch_idx):
        idx = batch['idx']
        rgb = batch['rgb']
        gt_depth = batch['depth']
        pred_depth = self.model(rgb)[0]

        out = [idx.cpu(), pred_depth.cpu(), gt_depth.cpu()]
        self.after_validation_step(out)


    # 20251214 【修改】重新定义 write_logger 类，以解决 20251210版本 在 Tensorboard 保存的图片灰蒙蒙的问题
    # 20251224 【修改】 
    '''
        原因：
          在训练过程中（SupervisedCombinedModel），RGB图片被转换成了数值范围大约在 -2.1 到 +2.6 之间的数据，而不是标准的 0 到 1。
          当 TensorBoard 接收到负数（例如 -1.5）时，它通常会将其截断为黑色 (0)。
          当数据分布被改变后，原本的颜色关系就被打乱了，导致你看上去是“灰蒙蒙”或者有些地方死黑。
        解决办法：
            在写入 Tensorboard 前进行“反归一化”：在 write_logger 函数中，把送入 Tensorboard 的图片先“还原”回 [0, 1] 的范围。
    '''
    def write_logger(self, loss_dict, rgb=None, pred=None, gt=None):
        """
        记录 Loss 和 图片到 TensorBoard / WandB
        """
        # 【20251224 新增】 和 SelfSupervisedTrainer.py 同样的计算逻辑
        steps_per_epoch = len(self.train_dataloader_obj)
        real_global_step = (self.epoch_offset * steps_per_epoch) + self.global_step
        '''
        # 20251214
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
            
            # --- 可视化逻辑 ---
            # 每隔 exp_freq (在 config 中定义，如 1000) 记录一次图片
            # 必须使用 .detach() 将 tensor 从计算图中分离，避免显存泄漏和影响训练
            if rgb is not None and self.global_step % self.config['exp_args']['exp_freq'] == 0:
                
                # 1. 处理 RGB (反归一化)
                # 取 batch 中的第一张图 [0]，形状变为 [3, H, W]
                # vis_rgb = self.denormalize(rgb[0].detach()).cpu()
                vis_rgb = rgb[0].detach().cpu().clamp(0, 1)
                self.logger.experiment.add_image('Train/RGB', vis_rgb, self.global_step)
                
                # 2. 处理 预测深度 (归一化显示)
                if pred is not None:
                    # clamp(0, 10) 限制显示范围，防止异常值破坏对比度
                    vis_pred = Tools.normalizeDepth(pred[0].detach().clamp(0, 10).cpu())
                    self.logger.experiment.add_image('Train/Pred', vis_pred, self.global_step)
                
                # 3. 处理 GT 深度 (归一化显示)
                if gt is not None:
                    vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())
                    self.logger.experiment.add_image('Train/GT', vis_gt, self.global_step)
            # -----------------
        '''
        # 20251224
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            # 【修改】 手动记录 Loss，使用 real_global_step
            if self.global_step % self.trainer.log_every_n_steps == 0:

                for key, val in loss_dict.items(): 
                    # self.log('Loss/%s'%key, val)  # 原作者
                    self.logger.experiment.add_scalar('Loss/%s'%key, val, real_global_step) # 20251225 修改
                    
                # --- 可视化逻辑 ---
                # 每隔 exp_freq (在 config 中定义，如 1000) 记录一次图片
                # 必须使用 .detach() 将 tensor 从计算图中分离，避免显存泄漏和影响训练
                if rgb is not None and self.global_step % self.config['exp_args']['exp_freq'] == 0:
                    
                    # 1. 处理 RGB (反归一化)
                    # 取 batch 中的第一张图 [0]，形状变为 [3, H, W]
                    # vis_rgb = self.denormalize(rgb[0].detach()).cpu()
                    vis_rgb = rgb[0].detach().cpu().clamp(0, 1)
                    # self.logger.experiment.add_image('Train/RGB', vis_rgb, self.global_step)
                    self.logger.experiment.add_image('Train/RGB', vis_rgb, real_global_step)    # 20251225 修改
                    
                    # 2. 处理 预测深度 (归一化显示)
                    if pred is not None:
                        # clamp(0, 10) 限制显示范围，防止异常值破坏对比度
                        vis_pred = Tools.normalizeDepth(pred[0].detach().clamp(0, 10).cpu())
                        # self.logger.experiment.add_image('Train/Pred', vis_pred, self.global_step)
                        self.logger.experiment.add_image('Train/Pred', vis_pred, real_global_step)  # 20251225 修改
                    
                    # 3. 处理 GT 深度 (归一化显示)
                    if gt is not None:
                        vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())
                        # self.logger.experiment.add_image('Train/GT', vis_gt, self.global_step)
                        self.logger.experiment.add_image('Train/GT', vis_gt, real_global_step)  # 20251225 修改
                # -----------------

        elif isinstance(self.logger, pl_loggers.WandbLogger):
            for key, val in loss_dict.items():
                self.log('Loss/%s'%key, val)
            # --- WandB 可视化逻辑 (与上面类似) ---
            if rgb is not None and self.global_step % self.config['exp_args']['exp_freq'] == 0:
                
                # vis_rgb = self.denormalize(rgb[0].detach()).cpu()
                # vis_pred = Tools.normalizeDepth(pred[0].detach().clamp(0, 10).cpu())
                # vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())
                vis_rgb = rgb[0].detach().cpu().clamp(0, 1)
                vis_pred = Tools.normalizeDepth(pred[0].detach().clamp(0, 10).cpu())
                vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())

                caption = ['RGB-%d'%(self.global_step), 'Pred-%d'%self.global_step, 'GT-%d'%self.global_step]
                self.logger.log_image('Figures', [vis_rgb, vis_pred, vis_gt], caption=caption)
            # -----------------------------------
        else:
            raise ValueError('Logger type weird')


    '''
    # 20251210 添加了旋转噪声，所以修改 write_logger：为了在 TensorBoard 中看到旋转后的图片
    # 同时，还要修改 training_step 来调用这个更新后的 write_logger
    def write_logger(self, loss_dict, rgb=None, pred=None, gt=None): # 修改参数签名
        """
        记录 Loss 和 图片到 TensorBoard / WandB
        """
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
            
            # --- 新增可视化代码 (参考自 SelfSupervisedTrainer) ---
            # 每 1000 step 记录一次图片，避免日志文件过大
            if rgb is not None and self.global_step % 1000 == 0:
                from .. import Tools # 确保引入了 Tools
                # 反归一化 RGB 以便显示
                # 注意：这里假设你的 Tools.normalizeDepth 已经存在
                vis_rgb = rgb[0].cpu() # 取 batch 中的第一张
                # 简单的反归一化 (粗略还原用于显示)
                vis_rgb = vis_rgb * 0.224 + 0.456 
                
                vis_pred = Tools.normalizeDepth(pred[0].detach().cpu())
                vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())
                
                self.logger.experiment.add_image('Train/RGB', vis_rgb, self.global_step)
                self.logger.experiment.add_image('Train/Pred', vis_pred, self.global_step)
                self.logger.experiment.add_image('Train/GT', vis_gt, self.global_step)
            # --------------------------------------------------

        elif isinstance(self.logger, pl_loggers.WandbLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
            # --- 新增可视化代码 (参考自 SelfSupervisedTrainer) ---
            # 每 1000 step 记录一次图片，避免日志文件过大
            if rgb is not None and self.global_step % 1000 == 0:
                from .. import Tools # 确保引入了 Tools
                # 反归一化 RGB 以便显示
                # 注意：这里假设你的 Tools.normalizeDepth 已经存在
                vis_rgb = rgb[0].cpu() # 取 batch 中的第一张
                # 简单的反归一化 (粗略还原用于显示)
                vis_rgb = vis_rgb * 0.224 + 0.456 
                
                vis_pred = Tools.normalizeDepth(pred[0].detach().cpu())
                vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())
                
                self.logger.experiment.add_image('Train/RGB', vis_rgb, self.global_step)
                self.logger.experiment.add_image('Train/Pred', vis_pred, self.global_step)
                self.logger.experiment.add_image('Train/GT', vis_gt, self.global_step)
            # --------------------------------------------------
        else:
            raise ValueError('Logger type weird')
    '''

    '''
    # 20251210 以前 原作者的代码
    def write_logger(self, loss_dict):
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
        elif isinstance(self.logger, pl_loggers.WandbLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
        else:
            raise ValueError('Logger type weird')
    '''
