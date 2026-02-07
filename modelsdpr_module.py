import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRModule(nn.Module):
    """Differentiable Physical Restoration Module based on Retinex prior"""
    
    def __init__(self, in_channels=3, hidden_dim=64):
        super(DPRModule, self).__init__()
        
        # 反射分量估计分支 (多尺度残差结构)
        self.reflectance_branch = nn.Sequential(
            DepthwiseSeparableResBlock(in_channels, hidden_dim),
            DepthwiseSeparableResBlock(hidden_dim, hidden_dim),
            DepthwiseSeparableResBlock(hidden_dim, in_channels),
            nn.Sigmoid()  # 输出范围[0,1]
        )
        
        # 光照分量估计分支 (全局池化+MLP)
        self.illumination_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels),
            nn.Sigmoid()
        )
        
        # 自适应校正因子
        self.gamma_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # 将gamma调整到[0.8, 1.2]范围
        self.gamma_scale = 0.4
        self.gamma_bias = 0.8
        
    def rgb_to_ycbcr(self, img):
        """RGB转YCbCr颜色空间"""
        r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat([y, cb, cr], dim=1)
    
    def ycbcr_to_rgb(self, img):
        """YCbCr转RGB颜色空间"""
        y, cb, cr = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
        cb = cb - 128
        cr = cr - 128
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1).clamp(0, 1)
    
    def forward(self, x):
        """
        输入: x [B, 3, H, W]
        输出: 恢复后的图像 [B, 3, H, W]
        """
        # 1. 颜色空间转换
        ycbcr = self.rgb_to_ycbcr(x)
        y_channel = ycbcr[:, 0:1, :, :]
        
        # 2. 估计初始反射分量
        r_init = self.reflectance_branch(y_channel)
        
        # 3. 估计光照分量
        l = self.illumination_branch(y_channel)
        l = l.view(-1, 1, 1, 1)  # 广播到全图
        
        # 4. 自适应校正因子
        gamma = self.gamma_net(y_channel)
        gamma = gamma * self.gamma_scale + self.gamma_bias
        
        # 5. 优化反射分量 (公式2)
        epsilon = 1e-6
        r_opt = gamma * r_init + epsilon
        
        # 6. 重建Y通道
        y_restored = r_opt * l
        
        # 7. 合并CbCr通道并转换回RGB
        ycbcr_restored = torch.cat([y_restored, ycbcr[:, 1:, :, :]], dim=1)
        restored_rgb = self.ycbcr_to_rgb(ycbcr_restored)
        
        return restored_rgb, r_opt, l

class DepthwiseSeparableResBlock(nn.Module):
    """深度可分离残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)