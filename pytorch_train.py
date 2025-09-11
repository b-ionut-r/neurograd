#!/usr/bin/env python3
"""
ImageNet-1k MobileNetV2 Training Script with PyTorch
Converted from NeuroGrad notebook with same hyperparameters and techniques.
"""

import os
import math
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ================================================
# Configuration
# ================================================
IMAGENET_DIR = Path('./imagenet')  # Change to your ImageNet directory
NUM_CLASSES = 1000
IMG_SHAPE = (224, 224)
LABEL_SMOOTH = 0.1
USE_ONE_HOT = True
BATCH_SIZE = 512
EPOCHS = 120
USE_AMP = True
EFFECTIVE_BATCH = BATCH_SIZE
ACCUM_STEPS = max(1, EFFECTIVE_BATCH // max(1, BATCH_SIZE))
WARMUP_EPOCHS = 5
MIN_LR = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'checkpoint.pth'


# ================================================
# Dataset and DataLoader Setup
# ================================================
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] logits
            target: [N, C] one-hot or [N] class indices
        """
        if target.dim() == 1:
            # Convert class indices to one-hot
            target = F.one_hot(target, num_classes=pred.size(-1)).float()
        
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -target * log_prob
        nll_loss = nll_loss.sum(dim=-1)
        
        if self.smoothing > 0:
            smooth_loss = -log_prob.mean(dim=-1)
            loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss.mean()


def to_one_hot(idx: int, num_classes: int = NUM_CLASSES, smooth: float = LABEL_SMOOTH):
    """Convert class index to one-hot with label smoothing."""
    v = torch.zeros(num_classes, dtype=torch.float32)
    if smooth and smooth > 0:
        off = smooth / num_classes
        v.fill_(off)
        v[idx] = 1.0 - smooth + off
    else:
        v[idx] = 1.0
    return v


class ImageNetDataset(Dataset):
    """Custom ImageNet dataset with label smoothing and one-hot encoding."""
    def __init__(self, root, transform=None, use_one_hot=True, label_smooth=0.1):
        self.dataset = ImageFolder(root)
        self.transform = transform
        self.use_one_hot = use_one_hot
        self.label_smooth = label_smooth
        self.num_classes = len(self.dataset.classes)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        if self.use_one_hot:
            label = to_one_hot(label, self.num_classes, self.label_smooth)
            
        return img, label


def create_data_loaders():
    """Create training and validation data loaders."""
    # Training augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SHAPE, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.05, 0.15), ratio=(0.3, 3.3))
    ])
    
    # Validation augmentations
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageNetDataset(
        root=IMAGENET_DIR / "train",
        transform=train_transform,
        use_one_hot=USE_ONE_HOT,
        label_smooth=LABEL_SMOOTH
    )
    
    val_dataset = ImageNetDataset(
        root=IMAGENET_DIR / "val", 
        transform=val_transform,
        use_one_hot=USE_ONE_HOT,
        label_smooth=0.0  # No smoothing for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.dataset.classes


# ================================================
# MobileNetV2 Model
# ================================================
def make_divisible(v: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Make number divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 architecture."""
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        
        # Inverted residual settings: [t, c, n, s]
        inverted_residual_setting = [
            [1, 16, 1, 1],   # t, c, n, s
            [6, 24, 2, 2],
            [6, 32, 3, 2], 
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        input_channel = make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        last_channel = make_divisible(1280 * max(1.0, width_mult), 8)
        
        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        # Building last several layers
        features.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        
        self.features = nn.Sequential(*features)
        
        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ================================================
# Training Utilities
# ================================================
def top_k_accuracies(y_true, y_pred, ks=(1, 3, 5)):
    """Calculate top-k accuracies."""
    with torch.no_grad():
        if y_true.dim() > 1:  # one-hot
            y_true = y_true.argmax(dim=-1)
        
        maxk = max(ks)
        batch_size = y_true.size(0)
        
        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))
        
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cosine_lr_schedule(step, total_steps, base_lr, min_lr, warmup_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def save_checkpoint(model, optimizer, scaler, epoch, val_acc, val_loss, global_step, path):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'global_step': global_step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scaler):
    """Load training checkpoint."""
    if not os.path.exists(path):
        return None, 1, -1.0, 0
    
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return (checkpoint, 
            checkpoint['epoch'] + 1, 
            checkpoint.get('val_acc', -1.0),
            checkpoint.get('global_step', 0))


def visualize_predictions(model, val_loader, classes, device, num_samples=4):
    """Visualize model predictions."""
    model.eval()
    
    # Get a random batch
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    with torch.no_grad():
        with autocast() if USE_AMP else nullcontext():
            outputs = model(images)
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    plt.figure(figsize=(15, 4))
    correct = 0
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        
        # Denormalize image for display
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        plt.imshow(img.permute(1, 2, 0))
        plt.axis('off')
        
        # Get predictions
        if USE_ONE_HOT:
            true_label = labels[idx].argmax().item()
        else:
            true_label = labels[idx].item()
        pred_label = outputs[idx].argmax().item()
        
        if true_label == pred_label:
            correct += 1
        
        true_class = classes[true_label] if true_label < len(classes) else str(true_label)
        pred_class = classes[pred_label] if pred_label < len(classes) else str(pred_label)
        
        plt.title(f'True: {true_class}\nPred: {pred_class}', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    print(f"Guessed {correct} / {num_samples}")


# ================================================
# Main Training Function  
# ================================================
def train_mobilenetv2():
    """Main training function."""
    print(f"Training on device: {DEVICE}")
    print(f"Using AMP: {USE_AMP}")
    
    # Create data loaders
    train_loader, val_loader, classes = create_data_loaders()
    steps_per_epoch = len(train_loader)
    opt_steps_per_epoch = (steps_per_epoch + ACCUM_STEPS - 1) // ACCUM_STEPS
    total_opt_steps = EPOCHS * opt_steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * opt_steps_per_epoch
    
    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Steps per epoch: {steps_per_epoch}, Optimization steps per epoch: {opt_steps_per_epoch}")
    
    # Create model
    model = MobileNetV2(num_classes=NUM_CLASSES, width_mult=1.0, dropout=0.2)
    model = model.to(DEVICE)
    
    # Create optimizer and loss function
    base_lr = 1e-3 * (BATCH_SIZE / 128.0)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    
    if USE_ONE_HOT:
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    
    # Setup mixed precision
    scaler = GradScaler() if USE_AMP else None
    
    # Load checkpoint if exists
    checkpoint, start_epoch, best_val_acc, global_opt_step = load_checkpoint(
        CHECKPOINT_PATH, model, optimizer, scaler)
    
    if checkpoint:
        print(f"Resumed from epoch {start_epoch-1}, best val acc: {best_val_acc:.4f}")
    
    print(f"\n[TRAIN] EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, "
          f"ACCUM_STEPS={ACCUM_STEPS}, BASE_LR={base_lr:.6f}")
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        start_time = time.time()
        
        running_loss = 0.0
        running_acc = 0.0
        running_top3 = 0.0  
        running_top5 = 0.0
        num_samples = 0
        micro_step = 0
        
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, 
                   desc=f"Epoch {epoch}/{EPOCHS}")
        
        for i, (images, targets) in pbar:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast() if USE_AMP else nullcontext():
                outputs = model(images)
                loss = criterion(outputs, targets) / ACCUM_STEPS
            
            # Backward pass
            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            micro_step += 1
            
            # Optimizer step with gradient accumulation
            if micro_step % ACCUM_STEPS == 0:
                # Update learning rate
                lr = cosine_lr_schedule(global_opt_step, total_opt_steps, base_lr, MIN_LR, warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                global_opt_step += 1
            
            # Statistics
            running_loss += loss.item() * ACCUM_STEPS
            
            with torch.no_grad():
                if USE_ONE_HOT:
                    y_true = targets.argmax(dim=-1)
                else:
                    y_true = targets
                
                top1, top3, top5 = top_k_accuracies(y_true, outputs, ks=(1, 3, 5))
                running_acc += top1.item() * len(images)
                running_top3 += top3.item() * len(images)
                running_top5 += top5.item() * len(images)
                num_samples += len(images)
            
            # Update progress bar
            if (i + 1) % 20 == 0:
                images_per_sec = num_samples / max(1e-9, time.time() - start_time)
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "lr": f"{current_lr:.5g}",
                    "loss": f"{running_loss / (i+1):.4f}",
                    "acc": f"{running_acc / num_samples:.1f}%",
                    "top3": f"{running_top3 / num_samples:.1f}%",
                    "top5": f"{running_top5 / num_samples:.1f}%",
                    "img/s": f"{images_per_sec:.1f}"
                })
        
        train_time = time.time() - start_time
        avg_train_loss = running_loss / steps_per_epoch
        avg_train_acc = running_acc / num_samples
        avg_train_top3 = running_top3 / num_samples  
        avg_train_top5 = running_top5 / num_samples
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[Epoch {epoch}] Train loss: {avg_train_loss:.4f} | "
              f"Train acc: {avg_train_acc:.2f}% | top3: {avg_train_top3:.2f}% | "
              f"top5: {avg_train_top5:.2f}% | time: {train_time:.1f}s | lr: {current_lr:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_top3 = 0.0
        val_top5 = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                
                with autocast() if USE_AMP else nullcontext():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * len(images)
                
                if USE_ONE_HOT:
                    y_true = targets.argmax(dim=-1)
                else:
                    y_true = targets
                
                top1, top3, top5 = top_k_accuracies(y_true, outputs, ks=(1, 3, 5))
                val_acc += top1.item() * len(images)
                val_top3 += top3.item() * len(images)
                val_top5 += top5.item() * len(images)
                val_samples += len(images)
        
        avg_val_loss = val_loss / val_samples
        avg_val_acc = val_acc / val_samples
        avg_val_top3 = val_top3 / val_samples
        avg_val_top5 = val_top5 / val_samples
        
        print(f"[Epoch {epoch}] Val   loss: {avg_val_loss:.4f} | "
              f"Val   acc: {avg_val_acc:.2f}% | top3: {avg_val_top3:.2f}% | "
              f"top5: {avg_val_top5:.2f}%")
        
        # Save best checkpoint
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_checkpoint(model, optimizer, scaler, epoch, 
                          avg_val_acc, avg_val_loss, global_opt_step, CHECKPOINT_PATH)
            print(f"✅ Saved checkpoint (val acc improved to {best_val_acc:.2f}%)")
        
        print("-" * 80)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for visualization
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model for visualization")
    
    # Visualize some predictions
    visualize_predictions(model, val_loader, classes, DEVICE, num_samples=4)


if __name__ == "__main__":
    # Verify dataset exists
    if not (IMAGENET_DIR / "train").exists() or not (IMAGENET_DIR / "val").exists():
        raise FileNotFoundError(f"Expected {IMAGENET_DIR}/train and {IMAGENET_DIR}/val to exist.")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train the model
    train_mobilenetv2()