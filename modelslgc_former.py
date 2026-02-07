import torch
import torch.nn as nn
import torch.nn.functional as F

class LGCFormer(nn.Module):
    """Local-Global Collaborative Transformer"""
    
    def __init__(self, in_channels=768, local_dim=256, global_dim=256, mamba_dim=512):
        super().__init__()
        
        # 特征对齐 (1x1卷积统一通道)
        self.channel_align = nn.ModuleList([
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1)
        ])
        
        # 注意力模块 (通道+空间注意力)
        self.attention = DualAttention(in_channels)
        
        # 局部分支 (动态卷积)
        self.local_branch = DynamicConvBranch(in_channels, local_dim)
        
        # 全局分支 (简化版Vision Mamba)
        self.global_branch = SimplifiedMamba(in_channels, global_dim, mamba_dim)
        
        # 门控融合
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 多尺度恢复
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(local_dim, 128, 3, stride=2, padding=1),
            nn.Conv2d(local_dim, 256, 3, stride=4, padding=1),
            nn.Conv2d(local_dim, 512, 3, stride=8, padding=1)
        ])
        
    def forward(self, features):
        """
        输入: features = [f3, f4, f5]
        输出: enhanced_features = [f3', f4', f5']
        """
        # 1. 特征对齐和融合
        aligned_features = []
        for i, (feat, align) in enumerate(zip(features, self.channel_align)):
            feat_aligned = align(feat)
            if i > 0:
                feat_aligned = F.interpolate(feat_aligned, size=features[0].shape[-2:], 
                                           mode='bilinear', align_corners=False)
            aligned_features.append(feat_aligned)
        
        # 拼接多尺度特征
        concat_feat = torch.cat(aligned_features, dim=1)
        
        # 2. 注意力增强
        attn_feat = self.attention(concat_feat)
        
        # 3. 双分支处理
        local_feat = self.local_branch(attn_feat)
        global_feat = self.global_branch(attn_feat)
        
        # 4. 门控融合 (公式3)
        gate = self.gate_net(attn_feat)
        fused_feat = gate * local_feat + (1 - gate) * global_feat
        
        # 5. 多尺度恢复和残差连接
        enhanced_features = []
        for i, downsample in enumerate(self.downsample_layers):
            if i == 0:
                feat_resized = fused_feat
            else:
                feat_resized = downsample(fused_feat)
            
            # 残差连接
            enhanced = feat_resized + features[i]
            enhanced_features.append(enhanced)
        
        return enhanced_features

class DualAttention(nn.Module):
    """双重注意力机制 (通道+空间)"""
    def __init__(self, channels):
        super().__init__()
        # 通道注意力 (SE模块)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_weights = self.channel_att(x)
        spatial_weights = self.spatial_att(x)
        return x * channel_weights * spatial_weights

class DynamicConvBranch(nn.Module):
    """动态卷积局部分支"""
    def __init__(self, in_channels, out_channels, kernel_size=3, num_kernels=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.group_size = out_channels // num_kernels
        
        # 生成动态卷积核的MLP
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_kernels * kernel_size * kernel_size)
        )
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1, groups=num_kernels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成动态卷积核
        kernels = self.kernel_generator(x)  # [B, num_kernels*k*k]
        kernels = kernels.view(B, self.num_kernels, self.kernel_size, self.kernel_size)
        
        # 应用动态卷积
        output = []
        for i in range(B):
            # 分组卷积
            group_outputs = []
            for g in range(self.num_kernels):
                start_ch = g * self.group_size
                end_ch = (g + 1) * self.group_size
                group_feat = x[i:i+1, start_ch:end_ch, :, :]
                kernel = kernels[i:i+1, g:g+1, :, :]
                conv_result = F.conv2d(group_feat, kernel, padding=self.kernel_size//2)
                group_outputs.append(conv_result)
            
            output_b = torch.cat(group_outputs, dim=1)
            output.append(output_b)
        
        return torch.cat(output, dim=0)

class SimplifiedMamba(nn.Module):
    """简化的Vision Mamba实现"""
    def __init__(self, in_channels, out_channels, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 状态空间参数
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, in_channels) * 0.01)
        self.C = nn.Parameter(torch.randn(out_channels, hidden_dim) * 0.01)
        
        # 投影层
        self.proj_in = nn.Linear(in_channels, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_channels)
        
        # 相似性加权矩阵
        self.similarity_weight = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 序列化
        x_seq = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
        
        # 投影到隐藏空间
        hidden = self.proj_in(x_seq)
        
        # 状态空间模型 (简化的线性递归)
        states = []
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        
        # 计算相似性权重
        sim_weights = self.similarity_weight(x_seq).squeeze(-1)  # [B, N]
        
        for t in range(x_seq.size(1)):
            # 加权状态更新
            weight_t = sim_weights[:, t:t+1].unsqueeze(-1)  # [B, 1, 1]
            input_t = x_seq[:, t:t+1, :]  # [B, 1, C]
            
            # 状态空间方程
            h = torch.matmul(h, self.A) + torch.matmul(input_t, self.B.t()) * weight_t
            state_t = torch.matmul(h, self.C.t())
            states.append(state_t)
        
        # 组合状态
        output_seq = torch.stack(states, dim=1)
        
        # 投影回特征空间
        output = self.proj_out(output_seq)
        
        # 恢复空间维度
        output = output.permute(0, 2, 1).view(B, -1, H, W)
        
        return output