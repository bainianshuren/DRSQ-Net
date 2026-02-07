import torch
import torch.nn as nn
import torch.nn.functional as F

class DSQHead(nn.Module):
    """Dynamic Sparse Query Detection Head"""
    
    def __init__(self, num_classes=4, num_queries=100, feature_dim=256, 
                 num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 可学习的对象查询
        self.object_queries = nn.Embedding(num_queries, feature_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, feature_dim, 1, 1))
        
        # 增强的交叉注意力
        self.cross_attn = EnhancedCrossAttention(feature_dim, num_heads, dropout)
        
        # 目标性评分模块
        self.objectness_scorer = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 任务感知动态路由
        self.routing_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim * 2),
        )
        
        # 分类分支
        self.cls_branch = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # 支持多标签
        )
        
        # 回归分支
        self.reg_branch = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),  # [x, y, w, h]
            nn.Sigmoid()  # 归一化坐标
        )
        
        self._init_parameters()
        
    def _init_parameters(self):
        # 初始化查询向量
        nn.init.uniform_(self.object_queries.weight, -0.1, 0.1)
        # 初始化位置编码
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
    def forward(self, features):
        """
        输入: features [B, C, H, W]
        输出: predictions [B, num_queries, num_classes+4+1]
        """
        B, C, H, W = features.shape
        
        # 1. 特征序列化
        features_seq = features.flatten(2)  # [B, C, N]
        N = features_seq.size(-1)
        
        # 2. 添加位置编码
        pos_enc = F.interpolate(self.pos_encoding, size=(H, W), mode='bilinear')
        pos_enc_seq = pos_enc.flatten(2)  # [1, C, N]
        features_with_pos = features_seq + pos_enc_seq
        
        # 3. 转置以进行注意力计算
        features_with_pos = features_with_pos.permute(0, 2, 1)  # [B, N, C]
        
        # 4. 获取对象查询
        queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, C]
        
        # 5. 增强的交叉注意力交互
        interacted_queries = self.cross_attn(queries, features_with_pos, features_with_pos)
        
        # 6. 任务感知动态路由
        routing_logits = self.routing_mlp(interacted_queries)  # [B, num_queries, 2C]
        routing_weights = torch.sigmoid(routing_logits)
        
        # 分割路由权重
        cls_weights = routing_weights[..., :self.feature_dim]
        reg_weights = routing_weights[..., self.feature_dim:]
        
        # 特征路由
        cls_features = interacted_queries * cls_weights
        reg_features = interacted_queries * reg_weights
        
        # 7. 分类和回归分支
        cls_scores = self.cls_branch(cls_features)  # [B, num_queries, num_classes]
        bbox_preds = self.reg_branch(reg_features)   # [B, num_queries, 4]
        
        # 8. 目标性评分
        objectness_scores = self.objectness_scorer(interacted_queries)  # [B, num_queries, 1]
        
        # 9. 组合预测结果
        predictions = torch.cat([cls_scores, bbox_preds, objectness_scores], dim=-1)
        
        return predictions, interacted_queries

class EnhancedCrossAttention(nn.Module):
    """增强的交叉注意力模块"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 线性投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 目标性评分权重 (公式4中的Ws)
        self.objectness_weight = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, query, key, value, objectness_scores=None):
        """
        输入: query [B, Nq, C], key/value [B, Nkv, C]
        输出: attended features [B, Nq, C]
        """
        B, Nq, C = query.shape
        Nkv = key.shape[1]
        
        # 线性投影
        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用目标性评分权重
        if objectness_scores is not None:
            obj_weights = objectness_scores.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Nkv]
            attn_scores = attn_scores * obj_weights
        
        # Softmax归一化
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # 注意力加权
        attn_output = torch.matmul(attn_probs, v)
        
        # 重组和输出投影
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, Nq, C)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        return attn_output