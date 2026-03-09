import os
import torch
from .BaseTrainer import BaseLitModule
from pytorch_lightning import loggers as pl_loggers
from .. import Loss
from .. import Tools 
# import BiFusev2.Loss as Loss
# import BiFusev2.Tools as Tools

class SupervisedLitModule_3D60(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.berHu = Loss.ReverseHuberLoss()
        # 【20260112 新增】初始化频域损失
        self.fft_loss = Loss.FrequencyDomainLoss()
        
        # # 20251214【新增】定义 ImageNet 的均值和标准差用于反归一化 (如果之前有，建议保留)
        # self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        # self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    # # 20251214【新增】反归一化 (辅助函数)
    # def denormalize(self, img_tensor):
    #     return img_tensor.mul(self.imagenet_std).add(self.imagenet_mean).clamp(0, 1)

    '''
    # 【20260115 新增】覆盖父类方法（直接覆盖 scheduler_args:下的参数），强制使用模型定义的分层优化器：使用分层学习策略
    def configure_optimizers(self):
        # 调用 BiFuse.py 中 SupervisedCombinedModel 定义的 configure_optimizers
        return self.model.configure_optimizers()
    '''
    
    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']
        gt_depth = batch['depth']
        pred_depth = self.model(rgb)
        
        # 1. 原有 BerHu Loss
        berhu_loss = self.berHu(pred_depth[0], gt_depth)
        
        # 2. 【20260112 新增】计算频域损失
        # 权重 lambda 可以调整，通常设为 0.1 或 0.05
        lambda_fft = 0.1 
        fft_loss = self.fft_loss(pred_depth[0], gt_depth)
        
        # 3. 【20260112 新增】总损失
        total_loss = berhu_loss + lambda_fft * fft_loss

        out = {
            'loss': total_loss,
            'berhu_loss': berhu_loss.detach(), # 记录分项 Loss 方便观察
            'fft_loss': fft_loss.detach()
        }

        # 调用 write_logger 并传入用于可视化的数据
        self.write_logger(out, rgb=rgb, pred=pred_depth[0], gt=gt_depth)
        
        return out
    

    def validation_step(self, batch, batch_idx):
        idx = batch['idx']
        rgb = batch['rgb']
        gt_depth = batch['depth']
        pred_depth = self.model(rgb)[0]

        out = [idx.cpu(), pred_depth.cpu(), gt_depth.cpu()]
        self.after_validation_step(out)


    def write_logger(self, loss_dict, rgb=None, pred=None, gt=None):
        """
        记录 Loss 和 图片到 TensorBoard / WandB
        """
        steps_per_epoch = len(self.train_dataloader_obj)
        real_global_step = (self.epoch_offset * steps_per_epoch) + self.global_step
        
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            # 【修改】 手动记录 Loss，使用 real_global_step
            if self.global_step % self.trainer.log_every_n_steps == 0:
                for key, val in loss_dict.items(): 
                    self.logger.experiment.add_scalar('Loss/%s'%key, val, real_global_step)
                    
            # --- 可视化逻辑 ---
            if rgb is not None and self.global_step % self.config['exp_args']['exp_freq'] == 0:
                
                # 1. 处理 RGB (反归一化)
                # 使用之前定义的反归一化参数，如果没有定义 denormalize 函数，可以使用简单 clamp
                # 但最好使用 denormalize 以获得正确的色彩
                vis_rgb = rgb[0].detach().cpu()
                if hasattr(self, 'imagenet_std'):
                     vis_rgb = vis_rgb.mul(self.imagenet_std.cpu()).add(self.imagenet_mean.cpu())
                vis_rgb = vis_rgb.clamp(0, 1)
                
                self.logger.experiment.add_image('Train/RGB', vis_rgb, real_global_step)
                
                # 2. 处理 预测深度
                if pred is not None:
                    vis_pred = Tools.normalizeDepth(pred[0].detach().clamp(0, 10).cpu())
                    self.logger.experiment.add_image('Train/Pred', vis_pred, real_global_step)
                
                # 3. 处理 GT 深度
                if gt is not None:
                    vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())
                    self.logger.experiment.add_image('Train/GT', vis_gt, real_global_step)

        elif isinstance(self.logger, pl_loggers.WandbLogger):
            for key, val in loss_dict.items():
                self.log('Loss/%s'%key, val)
            
            if rgb is not None and self.global_step % self.config['exp_args']['exp_freq'] == 0:
                vis_rgb = rgb[0].detach().cpu()
                if hasattr(self, 'imagenet_std'):
                     vis_rgb = vis_rgb.mul(self.imagenet_std.cpu()).add(self.imagenet_mean.cpu())
                vis_rgb = vis_rgb.clamp(0, 1)
                
                vis_pred = Tools.normalizeDepth(pred[0].detach().clamp(0, 10).cpu())
                vis_gt = Tools.normalizeDepth(gt[0].detach().cpu())

                caption = ['RGB-%d'%(self.global_step), 'Pred-%d'%self.global_step, 'GT-%d'%self.global_step]
                self.logger.log_image('Figures', [vis_rgb, vis_pred, vis_gt], caption=caption)
        else:
            raise ValueError('Logger type weird')