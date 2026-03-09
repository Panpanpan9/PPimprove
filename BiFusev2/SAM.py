# 20260106 修改返回值（最末尾）
import torch
import torch.nn as nn
import torch.nn.functional as F
# 直接使用 timm 的官方实现
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

'''
# --- 辅助函数 (替代 timm 依赖) ---
def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
'''    

# --- SAM 核心模块 ---

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, v_dim, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)) # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, v, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).view(B_, N, self.num_heads, -1).transpose(1, 2)
        kv = self.kv(v).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SAMBLOCK(nn.Module):
    """
    Args:
        dim (int): Number of feature channels
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 v_dim,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm

        self.norm1 = norm_layer(dim)
        self.normv = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, v_dim=v_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, v, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        shortcut_v = v
        v = self.normv(v)
        v = v.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # partition windows
        x_windows = window_partition(x, self.window_size) # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size) # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size,
                                   v_windows.shape[-1]) # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, v_windows, mask=None) # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.v_dim)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp) # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, self.v_dim)

        # FFN
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W


class SAM(nn.Module):
    def __init__(self,
                 input_dim=96,
                 embed_dim=96,
                 v_dim=64,
                 window_size=7,
                 num_heads=4,
                 patch_size=4,
                 in_chans=3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super().__init__()

        self.embed_dim = embed_dim

        if input_dim != embed_dim:
            self.proj_e = nn.Conv2d(input_dim, embed_dim, 3, padding=1)
        else:
            self.proj_e = None

        if v_dim != embed_dim:
            self.proj_q = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
        elif embed_dim % v_dim == 0:
            self.proj_q = None
        self.proj = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        v_dim = embed_dim
        self.sam_block = SAMBLOCK(
            dim=embed_dim,
            num_heads=num_heads,
            v_dim=v_dim,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=norm_layer)

        layer = norm_layer(embed_dim)
        layer_name = 'norm_sam'
        self.add_module(layer_name, layer)

    def forward(self, e, q):
        if self.proj_q is not None:
            q = self.proj_q(q)
        if self.proj_e is not None:
            e = self.proj_e(e)
        e_proj = e
        q_proj = q

        Wh, Ww = q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)
        e = e.flatten(2).transpose(1, 2)

        q_out, H, W = self.sam_block(q, e, Wh, Ww)
        norm_layer = getattr(self, f'norm_sam')
        q_out = norm_layer(q_out)
        q_out = q_out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        # return q_out + e_proj + q_proj    # 原 SAM 逻辑
        return q_out + q_proj   # 20260105 修改
        # 仅将 Attention 结果加到 Decoder 主流上：网络既保留了 Decoder 的语义(q_proj)，又融合了筛选后的 Encoder 细节(q_out)
        # 移除 + e_proj (原始Encoder特征)，只保留 Attention 筛选后的特征 + Decoder 主流


# ... (上面是 SAM.py 的已有代码) ...

if __name__ == '__main__':
    print("===== Testing SAM Module =====")
    
    # 模拟 BiFuse++ 中 Layer 3 的情况 (H/16 尺度)
    # 假设输入图像是 (512, 1024)，则 H/16 = 32, W/16 = 64
    # planes[1] 通常是 256 (对于 ResNet-18/34)
    B, C, H, W = 1, 256, 32, 64
    
    # 模拟 Encoder 特征 (Key/Value)
    e = torch.randn(B, C, H, W)
    # 模拟 Decoder 特征 (Query)
    q = torch.randn(B, C, H, W)
    
    # 实例化 SAM
    # input_dim=C (Encoder通道), v_dim=C (Decoder通道), embed_dim=C (内部计算通道)
    sam = SAM(input_dim=C, embed_dim=C, v_dim=C, window_size=7, num_heads=4)
    
    # 前向传播
    output = sam(e, q)
    
    print(f"Input Encoder shape: {e.shape}")
    print(f"Input Decoder shape: {q.shape}")
    print(f"Output SAM shape:    {output.shape}")
    
    # 验证逻辑
    if output.shape == q.shape:
        print("✅ Test Passed: Output shape matches Input Decoder shape.")
        # 注意：在原 BiFuse++ 中，这里使用的是 torch.cat([e, q], dim=1)，
        # 输出通道数会变成 2*C (512)。
        # 而 SAM 融合后通道数保持为 C (256)。
        # 这是符合预期的，因为我们在 BiFuse.py 中已经调整了后续卷积层的输入通道数。
        print("Note: Output channels are maintained (Fusion via Attention), not doubled (Fusion via Concat).")
    else:
        print("❌ Test Failed: Shape mismatch!")
        
# if __name__ == '__main__':

#     model = SAM(input_dim=96, embed_dim=96, v_dim=96, window_size=7, num_heads=4, patch_size=4, in_chans=3)

#     B, H, W = 3, 512, 1024
#     e = torch.rand(B, 96, H, W)
#     q = torch.rand(B, 96, H, W)

#     output = model(e, q)

#     print(e.size())
#     print(q.size())
#     print(output.size())