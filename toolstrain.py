import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.drsqnet import DRSQNet
from data.datasets import UnderwaterDataset
from utils.losses import HungarianMatcher
import yaml
import argparse

def train(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = DRSQNet(
        num_classes=config['model']['num_classes'],
        img_size=config['data']['img_size'],
        version=config['model']['version']
    ).to(device)
    
    # 数据加载
    train_dataset = UnderwaterDataset(
        root=config['data']['train_root'],
        img_size=config['data']['img_size'],
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            
            # 前向传播
            loss_dict = model(images, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            
            # 打印进度
            if batch_idx % config['training']['print_freq'] == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss_dict["total_loss"].item():.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存检查点
        if epoch % config['training']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader)
            }, f'checkpoints/drsqnet_epoch_{epoch}.pth')
        
        print(f'Epoch {epoch} completed. Average Loss: {total_loss/len(train_loader):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/drsqnet_small.yaml')
    args = parser.parse_args()
    
    train(args.config)