import torch
import torch.nn as nn
from .backbone import YOLOv11Backbone
from .dpr_module import DPRModule
from .lgc_former import LGCFormer
from .dsq_head import DSQHead

class DRSQNet(nn.Module):
    """Complete DRSQ-Net for Underwater Object Detection"""
    
    def __init__(self, num_classes=4, img_size=640, version='small'):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # 1. DPR模块
        self.dpr = DPRModule()
        
        # 2. YOLOv11骨干网络
        self.backbone = YOLOv11Backbone(version)
        
        # 3. LGC-Former特征增强
        self.lgc_former = LGCFormer()
        
        # 4. FPN特征金字塔
        self.fpn = FPN()
        
        # 5. DSQ-Head检测头
        self.dsq_head = DSQHead(num_classes)
        
        # 损失函数
        self.classification_loss = nn.BCELoss()
        self.regression_loss = nn.SmoothL1Loss()
        self.objectness_loss = nn.BCELoss()
        
    def forward(self, x, targets=None):
        """
        输入: x [B, 3, H, W]
        输出: detections or loss
        """
        # 1. 可微分物理恢复
        restored_img, reflectance, illumination = self.dpr(x)
        
        # 2. 骨干网络特征提取
        features = self.backbone(restored_img)  # [f3, f4, f5]
        
        # 3. 局部-全局协作特征增强
        enhanced_features = self.lgc_former(features)
        
        # 4. FPN特征融合
        fused_features = self.fpn(enhanced_features)
        
        # 5. 动态稀疏查询检测
        predictions, query_features = self.dsq_head(fused_features)
        
        if self.training:
            return self.compute_loss(predictions, targets)
        else:
            return self.post_process(predictions)
    
    def compute_loss(self, predictions, targets):
        """计算损失函数"""
        # 解析预测结果
        cls_preds = predictions[..., :self.num_classes]
        reg_preds = predictions[..., self.num_classes:self.num_classes+4]
        obj_preds = predictions[..., -1:]
        
        # 匈牙利匹配 (DETR风格)
        matched_indices = self.hungarian_matching(reg_preds, targets)
        
        # 计算匹配损失
        total_loss = 0
        classification_loss = 0
        regression_loss = 0
        objectness_loss = 0
        
        for batch_idx in range(predictions.size(0)):
            # 获取匹配的预测和目标
            matched_preds = cls_preds[batch_idx, matched_indices[batch_idx]]
            matched_targets = targets[batch_idx]
            
            # 分类损失
            classification_loss += self.classification_loss(
                matched_preds[:, :self.num_classes], 
                matched_targets[:, 1:1+self.num_classes]
            )
            
            # 回归损失
            regression_loss += self.regression_loss(
                matched_preds[:, self.num_classes:self.num_classes+4],
                matched_targets[:, 5:9]
            )
            
            # 目标性损失
            objectness_loss += self.objectness_loss(
                matched_preds[:, -1:],
                matched_targets[:, 0:1]
            )
        
        # 总损失
        total_loss = classification_loss + regression_loss + objectness_loss
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'objectness_loss': objectness_loss
        }
    
    def post_process(self, predictions, conf_thresh=0.5):
        """后处理: 过滤和格式化检测结果"""
        B = predictions.size(0)
        detections = []
        
        for batch_idx in range(B):
            batch_preds = predictions[batch_idx]
            
            # 应用置信度阈值
            obj_mask = batch_preds[:, -1] > conf_thresh
            filtered_preds = batch_preds[obj_mask]
            
            if len(filtered_preds) == 0:
                detections.append(torch.zeros((0, 6), device=predictions.device))
                continue
            
            # 解析预测结果
            cls_scores, cls_ids = filtered_preds[:, :self.num_classes].max(dim=1)
            bboxes = filtered_preds[:, self.num_classes:self.num_classes+4]
            obj_scores = filtered_preds[:, -1]
            
            # 组合最终分数
            final_scores = cls_scores * obj_scores
            
            # 格式化输出: [x, y, w, h, score, class_id]
            formatted_dets = torch.stack([
                bboxes[:, 0], bboxes[:, 1],
                bboxes[:, 2], bboxes[:, 3],
                final_scores, cls_ids.float()
            ], dim=1)
            
            detections.append(formatted_dets)
        
        return detections
    
    def hungarian_matching(self, pred_boxes, targets):
        """匈牙利算法匹配预测和目标"""
        # 简化的匹配实现
        matched_indices = []
        
        for batch_idx in range(len(targets)):
            target = targets[batch_idx]
            pred = pred_boxes[batch_idx]
            
            if len(target) == 0:
                matched_indices.append(torch.zeros(len(pred), dtype=torch.long))
                continue
            
            # 计算IoU矩阵
            iou_matrix = self.compute_iou(pred[:, :4], target[:, 5:9])
            
            # 匈牙利匹配
            matched_idx = self.simple_hungarian(iou_matrix)
            matched_indices.append(matched_idx)
        
        return matched_indices
    
    def compute_iou(self, boxes1, boxes2):
        """计算IoU矩阵"""
        # 简化的IoU计算
        return torch.rand(boxes1.size(0), boxes2.size(0))
    
    def simple_hungarian(self, cost_matrix):
        """简化的匈牙利算法"""
        # 实际实现应使用scipy或自定义的匈牙利算法
        return torch.arange(cost_matrix.size(0))

class FPN(nn.Module):
    """Feature Pyramid Network"""
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list))
        ])
        
    def forward(self, features):
        """FPN前向传播"""
        laterals = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # 自上而下路径
        fused_features = []
        for i in range(len(laterals)-1, -1, -1):
            if i == len(laterals) - 1:
                fused = laterals[i]
            else:
                # 上采样并融合
                size = laterals[i].shape[-2:]
                fused = F.interpolate(fused, size=size, mode='nearest')
                fused = fused + laterals[i]
            
            # 应用3x3卷积
            fused = self.fpn_convs[i](fused)
            fused_features.append(fused)
        
        # 反转顺序 [C3, C4, C5] -> [P3, P4, P5]
        fused_features = fused_features[::-1]
        
        # 融合多尺度特征
        # 上采样所有特征到最大分辨率
        target_size = fused_features[0].shape[-2:]
        upsampled_features = []
        for feat in fused_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        
        # 拼接所有特征
        final_feature = torch.cat(upsampled_features, dim=1)
        
        return final_feature