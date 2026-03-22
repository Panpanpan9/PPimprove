"""
Implementation of custom loss functions.

Notes:
- [TODO] masked L2 loss for sparse ground truth.
- [TODO] masked BerHu loss for sparse ground truth.
- [TODO] add docstring to CrossEntropyLoss.

Last update: 2018/11/05 by Johnson
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from .depth_level import *


def _assert_no_grad(tensor):
    """ To make sure tensor of ground truth cannot backprop """
    assert not tensor.requires_grad, \
               'nn criterions don\'t compute the gradient w.r.t. targets - please ' \
               'mark these tensors as not requiring gradients'


class L1Loss(nn.Module):
    """ L1 loss but mask out invalid points (value <= 0). """
    def __init__(self, dummy):
        super(L1Loss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach() #torch.Size([8, 1, 228, 304])
        diff = (targets - inputs)[valid_mask] #torch.Size([537908])
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = diff.abs().mean() #torch.Size([])
        import pdb;pdb.set_trace()
        return loss


class L2Loss(nn.Module):
    """ L2 loss but mask out invalid points (value <= 0). """
    def __init__(self, dummy):
        super(L2Loss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach()
        diff = (targets - inputs)[valid_mask]
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = diff.pow(2).mean()

        return loss

class HuberLoss(nn.Module):
    """Huber Loss but mask out invalid points (value <= 0). """
    def __init__(self, dummy):
        super(HuberLoss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach() #Pooling if >0 => True => 1 
        diff = (targets - inputs)[valid_mask]
        loss_fn = nn.SmoothL1Loss(reduce = False, size_average = False)
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = loss_fn(inputs, targets)

        return loss
class Berhu_uncertainty(nn.Module):
    def __init__(self):
        super(Berhu_uncertainty, self).__init__()
    def forward(self, inputs, targets, un):
        _assert_no_grad(targets) #cannot backprop the gradient
        valid_mask = (targets > 0).detach().float()
        diff = (targets - inputs) * valid_mask
        adiff = torch.abs(targets - inputs) * valid_mask
        MAX = adiff.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0] 
        c = 0.2 * MAX
        sqdiff = (diff * diff + c * c) / (2 * c + 1e-6 )
        b_loss1 = adiff * (adiff <= c).cuda().float()
        b_loss2 = sqdiff * (adiff > c).cuda().float()
        b_loss = b_loss1 + b_loss2
        #print(b_loss.max(), b_loss.min())
        #import pdb;pdb.set_trace()
        u_loss1 = torch.exp(-(un.clamp(min=-5))) * b_loss
        u_loss2 = un
        u_loss = ((0.5 * u_loss1 + 0.5 * u_loss2) * valid_mask).sum() / valid_mask.sum()
        return u_loss, (u_loss1*valid_mask).sum()/valid_mask.sum(), (u_loss2[valid_mask.byte().detach()]).mean(), (b_loss[valid_mask.byte().detach()]).mean()

class Berhu_un_2branch(nn.Module):
    def __init__(self):
        super(Berhu_un_2branch,self).__init__()
    def forward(self, d_equi, d_c2e, targets, equi_un, c2e_un):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach().float()
        diff_e = (targets - d_equi) * valid_mask
        adiff_e = torch.abs(targets - d_equi) * valid_mask
        MAX_e = adiff_e.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0]
        c_e = 0.2 * MAX_e
        
        diff_c = (targets - d_c2e) * valid_mask
        adiff_c = torch.abs(targets - d_c2e) * valid_mask
        MAX_c = adiff_c.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0]
        c_c = 0.2 * MAX_c

        sqdiff_e = (diff_e * diff_e + c_e * c_e) / (2 * c_e + 1e-6 )
        b_loss1_e = adiff_e * (adiff_e <= c_e).cuda().float()
        b_loss2_e = sqdiff_e * (adiff_e > c_e).cuda().float()
        b_loss_e = b_loss1_e + b_loss2_e

        sqdiff_c = (diff_c * diff_c + c_c * c_c) / (2 * c_c + 1e-6)
        b_loss1_c = adiff_c * (adiff_c <= c_c).cuda().float()
        b_loss2_c = sqdiff_c * (adiff_c > c_c).cuda().float()
        b_loss_c = b_loss1_c + b_loss2_c 

        merge_b = b_loss_e.clone()
        merge_b[equi_un > c2e_un] = b_loss_c[equi_un > c2e_un]

        #method_1
        merge_un = equi_un.clone()
        merge_un[equi_un > c2e_un] = c2e_un[equi_un > c2e_un]
        merge_un_bad = equi_un.clone()
        merge_un_bad[equi_un < c2e_un] = c2e_un[equi_un < c2e_un]
        #method_2
        #merge_mask = torch.where(equi_un < cube_un, equi_un, cube_un)
        
        u_loss1 = torch.exp(-(merge_un.clamp(min=-5))) * merge_b
        u_loss2 = merge_un
        u_loss3 = merge_un_bad
        u_loss = ((0.5 * u_loss1 + 0.5 * u_loss2 + 0.05 * u_loss3) * valid_mask).sum() / valid_mask.sum()
        return u_loss, (u_loss1*valid_mask).sum()/valid_mask.sum(),(u_loss2[valid_mask.byte().detach()]).mean(), (merge_b[valid_mask.byte().detach()]).mean()

#NCHW
class ReverseHuberLoss(nn.Module):
    def __init__(self, dummy=None):
        super(ReverseHuberLoss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets) #cannot backprop the gradient
        # print("-------- SupervisedLoss.py中 ReverseHuberLoss类中的forward函数中: --------")
        valid_mask = (targets > 0).detach() #Pooling if >0 => True => 1    torch.Size([8, 1, 228, 304])
        # print("valid_mask.shape:", valid_mask.shape)   # valid_mask.shape: torch.Size([8, 1, 228, 304])
        
        valid_mask = valid_mask.float() #uint8 to float32 torch.Size([8, 1, 228, 304])
        # print("valid_mask.shape:", valid_mask.shape)   # valid_mask.shape: torch.Size([8, 1, 228, 304])
        
        diff = (targets - inputs) * valid_mask #torch.Size([8, 1, 228, 304])
        # print("diff.shape:", diff.shape)   # diff.shape: torch.Size([8, 1, 228, 304])

        adiff = torch.abs(targets - inputs) * valid_mask #torch.Size([8, 1, 228, 304])
        # print("adiff.shape:", adiff.shape)   # adiff.shape: torch.Size([8, 1, 228, 304])
        
        MAX = adiff.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0] #torch.Size([8, 1, 1, 1])
        # print("MAX.shape:", MAX.shape)   # MAX.shape: torch.Size([8, 1, 1, 1])
        
        c = 0.2 * MAX #torch.Size([8, 1, 1, 1])
        sqdiff = (diff * diff + c * c) / (2 * c + 1e-6 )  #torch.Size([8, 1, 228, 304])
        # print("sqdiff.shape:", sqdiff.shape)   # sqdiff.shape: torch.Size([8, 1, 228, 304])
        
        loss1 = adiff * (adiff <= c).cuda().float() #torch.Size([8, 1, 228, 304])
        loss2 = sqdiff * (adiff > c).cuda().float() #torch.Size([8, 1, 228, 304])
        # print("loss1.shape:", loss1.shape)   # loss1.shape: torch.Size([8, 1, 228, 304])
        # print("loss2.shape:", loss2.shape)   # loss2.shape: torch.Size([8, 1, 228, 304])
        
        loss = ((loss1 + loss2)  / (valid_mask.sum() + 1e-6)).sum()
        # print("loss.shape:", loss.shape)   # loss.shape: torch.Size([])
        
        
        return loss

class CrossEntropyLoss(nn.Module):
    """ Cross entropy loss with logits and regression targets. """
    def __init__(self, depth_level, min_depth, max_depth, n_D):
        super(CrossEntropyLoss, self).__init__()
        # Setup depth level
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_D = n_D
        if depth_level == 'space_increasing':
            self.depth_level = space_increasing_depth_level(self.min_depth,
                                                            self.max_depth,
                                                            self.n_D)
        else:
            raise ValueError('Invalid depth level {}'.format(self.depth_level))
        self.depth_level = np.array(self.depth_level)

        # Setup criterion
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach().squeeze(1)

        # Change regression targets to classification
        tiled_targets = targets.repeat(1, self.n_D, 1, 1)
        broadcast_d_1 = torch.Tensor(self.depth_level[np.newaxis, :, 0:1, np.newaxis]).to(targets)
        broadcast_d_2 = torch.Tensor(self.depth_level[np.newaxis, :, 1:2, np.newaxis]).to(targets)
        targets_cls = (tiled_targets >= broadcast_d_1) * (tiled_targets < broadcast_d_2)
        _, targets_cls = targets_cls.max(1)

        # Compute loss
        loss = self.criterion(inputs, targets_cls.long())
        loss = loss[valid_mask].mean()

        return loss


class NStageLoss(nn.Module):
    """ N-stage (multi-scale) loss. Target is only the one in the last stage. """
    def __init__(self, n_stages, interp, weights, criterion):
        super(NStageLoss, self).__init__()
        assert len(weights) == n_stages, 'The length of weights should be the \
                                          same as number of stages.'
        self.n_stages = n_stages
        self.interp = interp
        self.weights = weights
        if criterion == 'L1Loss':
            self.criterion = L1Loss(None)
        elif criterion == 'L2Loss':
            self.criterion = L2Loss(None)
        else:
            raise ValueError('Invalid criterion {}'.format(criterion))

    def forward(self, inputs, targets):
        _assert_no_grad(targets)

        total_loss = 0.
        for i in range(self.n_stages):
            input_i = inputs[i]
            if i == (self.n_stages - 1):
                target_i = targets
            else:
                target_i = F.interpolate(targets, input_i.shape[2:], mode=self.interp)
            loss = self.criterion(input_i, target_i)
            total_loss += self.weights[i] * loss
            print(loss)

        return total_loss


class NStageCrossEntropyLoss(nn.Module):
    """ N-stage (multi-scale) Cross Entropy loss. Target is only the one in the last stage. """
    def __init__(self, n_stages, interp, weights, depth_level, min_depth, max_depth, n_D):
        super(NStageCrossEntropyLoss, self).__init__()
        assert len(weights) == n_stages, 'The length of weights should be the \
                                          same as number of stages.'
        self.n_stages = n_stages
        self.interp = interp
        self.weights = weights
        self.criterion = CrossEntropyLoss(depth_level, min_depth, max_depth, n_D)

    def forward(self, inputs, targets):
        _assert_no_grad(targets)

        total_loss = 0.
        for i in range(self.n_stages):
            input_i = inputs[i]
            if i == (self.n_stages - 1):
                target_i = targets
            else:
                target_i = F.interpolate(targets, input_i.shape[2:], mode=self.interp)
            loss = self.criterion(input_i, target_i)
            total_loss += self.weights[i] * loss
            print(loss)

        return total_loss


# =============================================================================
# 20260318 改进版频域损失函数 (FreDSNet 核心思想)
# =============================================================================

class FrequencyDomainLoss(nn.Module):
    """
    多尺度频域损失函数 - 将频域感知提炼为优化约束

    功能特性:
    1. 多尺度频域约束: 在不同分辨率下计算频域一致性
    2. 多种损失类型: 幅度谱、相位谱、完整复数
    3. 频率加权: 可选择关注低频/高频/全频
    4. 智能 Mask 处理: 避免 FFT 边界效应

    参数:
        loss_type (str): 'magnitude' | 'phase' | 'complex' | 'all'
        freq_weight (str): 'low' | 'high' | 'all' - 频率加权策略
        alpha (float): 频域损失权重 (默认 0.1，从小开始)
        multiscale (bool): 是否启用多尺度 (默认 True)
        num_scales (int): 多尺度数量 (默认 3)
    """

    def __init__(self,
                 loss_type='magnitude',
                 freq_weight='all',
                 alpha=0.1,
                 multiscale=True,
                 num_scales=3):
        super(FrequencyDomainLoss, self).__init__()
        self.loss_type = loss_type
        self.freq_weight = freq_weight
        self.alpha = alpha
        self.multiscale = multiscale
        self.num_scales = num_scales

    def _generate_frequency_weight(self, H, W, device):
        """生成频率权重矩阵"""
        # rfft2 输出形状: [H, W//2 + 1]
        freq_h = torch.fft.fftfreq(H, device=device).abs()
        freq_w = torch.fft.rfftfreq(W, device=device).abs()

        # 2D 频率网格
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)

        if self.freq_weight == 'low':
            # 低频加权 (高斯衰减)
            weight = torch.exp(-2 * freq_grid**2)
        elif self.freq_weight == 'high':
            # 高频加权 (边缘增强)
            weight = 1 - torch.exp(-2 * freq_grid**2)
        else:  # 'all'
            weight = torch.ones_like(freq_grid)

        return weight[None, None, :, :]  # [1, 1, H, W//2+1]

    def _compute_fft_loss_single_scale(self, pred, gt, freq_weight=None):
        """单尺度频域损失计算"""
        # 1. Mask 处理：将无效值设为 0
        valid_mask = (gt > 0).float()
        pred_masked = pred * valid_mask
        gt_masked = gt * valid_mask

        # 2. FFT 变换
        fft_pred = torch.fft.rfft2(pred_masked)
        fft_gt = torch.fft.rfft2(gt_masked)

        # 3. 计算不同类型的损失
        if self.loss_type == 'magnitude':
            # 幅度谱损失 (强度信息)
            mag_pred = torch.abs(fft_pred)
            mag_gt = torch.abs(fft_gt)
            loss = F.l1_loss(mag_pred, mag_gt)

        elif self.loss_type == 'phase':
            # 相位谱损失 (结构/位置信息)
            phase_pred = torch.angle(fft_pred)
            phase_gt = torch.angle(fft_gt)
            loss = F.l1_loss(phase_pred, phase_gt)

        elif self.loss_type == 'complex':
            # 复数损失 (实部 + 虚部)
            loss_real = F.l1_loss(torch.real(fft_pred), torch.real(fft_gt))
            loss_imag = F.l1_loss(torch.imag(fft_pred), torch.imag(fft_gt))
            loss = loss_real + loss_imag

        elif self.loss_type == 'all':
            # 组合损失
            mag_loss = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_gt))
            phase_loss = F.l1_loss(torch.angle(fft_pred), torch.angle(fft_gt))
            loss = mag_loss + 0.5 * phase_loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

    def forward(self, inputs, targets):
        """
        计算预测深度和真实深度在频域上的差异

        Args:
            inputs: [B, 1, H, W] 预测深度
            targets: [B, 1, H, W] 真实深度

        Returns:
            loss: 标量损失值
        """
        _assert_no_grad(targets)

        if not self.multiscale:
            # 单尺度
            freq_loss = self._compute_fft_loss_single_scale(inputs, targets)
            return self.alpha * freq_loss
        else:
            # 多尺度频域损失
            total_loss = 0.0
            B, C, H, W = inputs.shape

            for i in range(self.num_scales):
                scale_factor = 2 ** i
                if i == 0:
                    pred_scaled = inputs
                    gt_scaled = targets
                else:
                    new_h, new_w = H // scale_factor, W // scale_factor
                    pred_scaled = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    gt_scaled = F.interpolate(targets, size=(new_h, new_w), mode='bilinear', align_corners=False)

                # 计算该尺度的频域损失
                scale_loss = self._compute_fft_loss_single_scale(pred_scaled, gt_scaled)
                total_loss += scale_loss

            # 平均并应用权重
            return self.alpha * total_loss / self.num_scales


class FrequencyConsistencyLoss(nn.Module):
    """
    频域一致性损失 - 轻量级版本

    相比 FrequencyDomainLoss 更简单，计算更快，适合作为辅助损失

    参数:
        alpha (float): 损失权重 (默认 0.05)
        use_magnitude (bool): 是否使用幅度谱 (默认 True)
        use_phase (bool): 是否使用相位谱 (默认 True)
    """

    def __init__(self, alpha=0.05, use_magnitude=True, use_phase=True):
        super(FrequencyConsistencyLoss, self).__init__()
        self.alpha = alpha
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase

    def forward(self, inputs, targets):
        """
        计算频域一致性损失

        Args:
            inputs: [B, 1, H, W] 预测深度
            targets: [B, 1, H, W] 真实深度

        Returns:
            loss: 标量损失值
        """
        _assert_no_grad(targets)

        # Mask 处理
        valid_mask = (targets > 0).float()
        inputs_masked = inputs * valid_mask
        targets_masked = targets * valid_mask

        # FFT 变换
        fft_pred = torch.fft.rfft2(inputs_masked)
        fft_gt = torch.fft.rfft2(targets_masked)

        total_loss = 0.0

        if self.use_magnitude:
            # 幅度谱损失
            mag_pred = torch.abs(fft_pred)
            mag_gt = torch.abs(fft_gt)
            total_loss += F.l1_loss(mag_pred, mag_gt)

        if self.use_phase:
            # 相位谱损失
            phase_pred = torch.angle(fft_pred)
            phase_gt = torch.angle(fft_gt)
            total_loss += F.l1_loss(phase_pred, phase_gt)

        return self.alpha * total_loss


class CombinedLossWithFrequency(nn.Module):
    """
    组合损失函数 - 空域损失 + 频域损失

    将传统的 ReverseHuberLoss 与频域损失结合，提供更全面的约束

    参数:
        spatial_weight (float): 空域损失权重 (默认 1.0)
        frequency_weight (float): 频域损失权重 (默认 0.1，从小开始)
        frequency_loss_type (str): 'magnitude' | 'phase' | 'complex' | 'all'
        use_multiscale_freq (bool): 是否使用多尺度频域损失
        verbose (bool): 是否打印详细损失信息
    """

    def __init__(self,
                 spatial_weight=1.0,
                 frequency_weight=0.1,
                 frequency_loss_type='magnitude',
                 use_multiscale_freq=False,
                 verbose=False):
        super(CombinedLossWithFrequency, self).__init__()

        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        self.verbose = verbose

        # 空域损失 (使用原有的 ReverseHuberLoss)
        self.spatial_loss = ReverseHuberLoss(None)

        # 频域损失
        if use_multiscale_freq:
            self.frequency_loss = FrequencyDomainLoss(
                loss_type=frequency_loss_type,
                alpha=1.0,  # 权重在外部控制
                multiscale=True,
                num_scales=3
            )
        else:
            # 使用轻量级版本
            self.frequency_loss = FrequencyConsistencyLoss(
                alpha=1.0,  # 权重在外部控制
                use_magnitude=(frequency_loss_type in ['magnitude', 'all']),
                use_phase=(frequency_loss_type in ['phase', 'all'])
            )

    def forward(self, inputs, targets):
        """
        计算组合损失

        Args:
            inputs: [B, 1, H, W] 预测深度
            targets: [B, 1, H, W] 真实深度

        Returns:
            total_loss: 总损失
        """
        _assert_no_grad(targets)

        # 空域损失
        loss_spatial = self.spatial_loss(inputs, targets)

        # 频域损失
        loss_frequency = self.frequency_loss(inputs, targets)

        # 总损失
        total_loss = (self.spatial_weight * loss_spatial +
                      self.frequency_weight * loss_frequency)

        # 打印详细信息
        if self.verbose:
            print(f"Loss - Spatial: {loss_spatial.item():.4f}, "
                  f"Frequency: {loss_frequency.item():.4f}, "
                  f"Total: {total_loss.item():.4f}")

        return total_loss

    def get_loss_components(self, inputs, targets):
        """
        获取各项损失的详细值，用于 TensorBoard 记录

        Returns:
            dict: {'spatial_loss': float, 'frequency_loss': float, 'total_loss': float}
        """
        _assert_no_grad(targets)

        loss_spatial = self.spatial_loss(inputs, targets)
        loss_frequency = self.frequency_loss(inputs, targets)
        total_loss = (self.spatial_weight * loss_spatial +
                      self.frequency_weight * loss_frequency)

        return {
            'spatial_loss': loss_spatial.item(),
            'frequency_loss': loss_frequency.item(),
            'total_loss': total_loss.item()
        }


# =============================================================================
# 使用示例和配置建议
# =============================================================================

"""
【配置建议】在 config.yaml 中添加以下配置:

loss_args:
  # 基础配置 (推荐从这些值开始)
  use_frequency_loss: true          # 是否启用频域损失
  frequency_loss_weight: 0.05       # 频域损失权重 (从小开始，0.01-0.1)
  frequency_loss_type: 'magnitude'  # 'magnitude'(推荐) | 'phase' | 'complex' | 'all'
  use_multiscale_frequency: false   # 是否启用多尺度 (建议先关闭)

  # 进阶配置 (在基础配置稳定后再尝试)
  # frequency_loss_weight: 0.1      # 逐步增加
  # use_multiscale_frequency: true  # 启用多尺度
  # frequency_num_scales: 3         # 多尺度数量


【使用方式 1】在 Trainer 中直接使用:

from BiFusev2.Loss.SupervisedLoss import CombinedLossWithFrequency

# 创建损失函数
criterion = CombinedLossWithFrequency(
    spatial_weight=1.0,
    frequency_weight=0.05,  # 从小开始
    frequency_loss_type='magnitude',
    use_multiscale_freq=False,
    verbose=True  # 训练时查看详细损失
)

# 计算 loss
loss = criterion(pred_depth, gt_depth)


【使用方式 2】渐进式启用 (推荐):

# 第一阶段：仅空域训练 (baseline)
criterion = ReverseHuberLoss(None)

# 第二阶段：加入轻量级频域约束
criterion = CombinedLossWithFrequency(
    spatial_weight=1.0,
    frequency_weight=0.01,  # 很小的权重
    frequency_loss_type='magnitude',
    use_multiscale_freq=False
)

# 第三阶段：逐步增加频域权重
frequency_weights = [0.01, 0.03, 0.05, 0.1]


【超参数调整指南】:
1. frequency_loss_weight:
   - 初始值: 0.01 ~ 0.05
   - 空域损失通常在 0.1~1.0 范围，频域损失较小，所以权重要小
   - 如果训练不稳定，降低权重

2. frequency_loss_type:
   - 'magnitude': 幅度谱 (推荐优先尝试)
   - 'phase': 相位谱 (更关注结构)
   - 'all': 组合 (可能过于约束)

3. use_multiscale_frequency:
   - False: 单尺度，计算快
   - True: 多尺度，更全面但计算量增加 ~2-3x

4. 预期效果:
   - 更好的全局结构一致性
   - 对噪声更鲁棒
   - 指标提升: RMSE ↓, MAE ↓, δ1 ↑ (预期 1-3%)
"""