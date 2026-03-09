import os
import torch
from pytorch_lightning import loggers as pl_loggers
from .BaseTrainer import BaseLitModule
from .. import Loss, Tools

# Trainer 初始化: 在父类 SelfSupervisedLitModule 的初始化函数中，默认实例化了 ContrastPhotometric (即 CAPL)。
class SelfSupervisedLitModule(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 这里默认初始化了 CAPL 模块（SPL 则是在 main.py中 手动将 photo_loss 覆盖为了 BasePhotometric()-> self.photo_loss = BiFusev2.Loss.BasePhotometric()）
        self.photo_loss = Loss.ContrastPhotometric()

    # 计算（两个正则化）: 在 training_step 中调用 self.photo_loss(...) 时，实际上执行的就是 ContrastPhotometric.forward()
    def training_step(self, batch, batch_idx):
        ref = batch['ref']
        tgts = batch['tgts']
        pose = batch['pose']
        gt_depth = batch['depth']

        pred_depth, pred_mask, pred_pose = self.model(ref, tgts)
        pred_mask = [x.clamp(0, 1) for x in pred_mask]

        # 1. 计算光度损失 (这里是 CAPL 或 SPL，取决于 self.photo_loss 是什么)
        rec_loss, _ = self.photo_loss(ref, pred_depth, pred_mask, tgts, pred_pose)
        # 2. 调用 Mask 正则化项 (Explainability Loss)
        # 对应论文公式 (7): 防止 Mask 全为 0
        exp_loss = Loss.explainability_loss(pred_mask)
        # 3. 调用 平滑正则化项 (Smooth Loss)
        # 对应论文公式 (8): 保证深度图平滑
        smooth_loss = Loss.smooth_loss(pred_depth)
        # 4. 最终的损失：总 Loss 加权求和
        loss = rec_loss + 0.1 * exp_loss + 0.01 * smooth_loss

        out = {
            'loss': loss,
            'rec-loss': rec_loss,
            'exp-loss': exp_loss,
            'smooth-loss': smooth_loss
        }
        self.write_logger(out, ref, pred_depth, gt_depth)
        
        return out
    
    def validation_step(self, batch, batch_idx):
        idx = batch['idx']
        ref = batch['ref']
        tgts = batch['tgts']
        pose = batch['pose']
        gt_depth = batch['depth']

        pred_depth, pred_mask, pred_pose = self.model(ref, tgts)
        out = [idx.cpu(), pred_depth[0].cpu(), gt_depth.cpu()]
        self.after_validation_step(out)


    '''
    # 原作者的   
    def write_logger(self, loss_dict, ref, pred_depth, gt_depth):     
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
            if self.global_step % self.config['exp_args']['exp_freq'] == 0:
                rgb = ref.data.cpu()
                pred_depth = Tools.normalizeDepth(pred_depth[0].clamp(0, 10).data.cpu())
                gt_depth = Tools.normalizeDepth(gt_depth.data.cpu())

                self.logger.experiment.add_images('RGB', rgb, self.global_step)
                self.logger.experiment.add_images('Depth/Pred', pred_depth, self.global_step)
                self.logger.experiment.add_images('Depth/GT', gt_depth, self.global_step)
    '''
    # 【20251224 修改】 使得 TenserBoard 根据是否加载模型 调整起止的epoch
    def write_logger(self, loss_dict, ref, pred_depth, gt_depth):
        
        # 【新增】 1. 计算 当前真实的全局步数 (Global Step)
        # self.train_dataloader_obj 是在训练开始后生成的，此时可以使用 len() 获取长度
        steps_per_epoch = len(self.train_dataloader_obj)
        # 真实的全局步数 = 之前跳过的步数 (offset * 每轮步数) + 当前新跑的步数
        real_global_step = (self.epoch_offset * steps_per_epoch) + self.global_step

        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            # 【修改 1】 训练 Loss：手动使用 real_global_step 记录     
            # 注意：为了防止日志文件过大，我们通常遵循 log_every_n_steps (默认50)
            if self.global_step % self.trainer.log_every_n_steps == 0:
                for key, val in loss_dict.items():
                    self.logger.experiment.add_scalar('Loss/%s'%key, val, real_global_step)

            # 【修改 2】 图像可视化：使用 real_global_step
            if self.global_step % self.config['exp_args']['exp_freq'] == 0:
                rgb = ref.data.cpu()
                pred_depth = Tools.normalizeDepth(pred_depth[0].clamp(0, 10).data.cpu())
                gt_depth = Tools.normalizeDepth(gt_depth.data.cpu())

                #  使用 real_global_step
                self.logger.experiment.add_images('RGB', rgb, real_global_step)
                self.logger.experiment.add_images('Depth/Pred', pred_depth, real_global_step)
                self.logger.experiment.add_images('Depth/GT', gt_depth, real_global_step)
        
        elif isinstance(self.logger, pl_loggers.WandbLogger):
            # 【20251224 修改】WandB 的逻辑同理 懒得修改了
            for key, val in loss_dict.items(): 
                self.log('Loss/%s'%key, val)
            if self.global_step % self.config['exp_args']['exp_freq'] == 0:
                rgb = ref.data.cpu()
                pred_depth = Tools.normalizeDepth(pred_depth[0].clamp(0, 10).data.cpu())
                gt_depth = Tools.normalizeDepth(gt_depth.data.cpu())
                caption = ['RGB-%d'%(self.global_step), 'Pred-%d'%self.global_step, 'GT-%d'%self.global_step]
                self.logger.log_image('Figures', [rgb, pred_depth, gt_depth], caption=caption)
        else:
            raise ValueError('logger type weird')
        