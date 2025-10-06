import os
import random
import json
import cv2
import numpy as np
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Calculate class weights to handle data imbalance => adjust weights in the loss function so that the model pays more attention to minority classes
def compute_class_weights(labels, num_classes):
    cnt = np.bincount(labels, minlength=num_classes) # count frequency occurrence of each class
    cnt[cnt == 0] = 1 # avoid division by zero
    weights = 1.0 / cnt
    nor_weights = weights * (num_classes / np.sum(weights)) # normalize weights to num_classes
    return torch.tensor(nor_weights, dtype=torch.float32)



# =================================================================================================
# Increase the diversity of training data
# =================================================================================================

# Create smoothed weights using Exponential Moving Average (EMA)
# + Reduce noise in weight updates
# + Help model converge better and avoid overfitting
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    # Sau các bước cập nhật trọng số => param.data thay đổi => shadow cũng thay đổi để làm mượt
    # Ví dụ trong quá trình train (param=0.6, shadow=0.5) => param.data từ 0.6 => 0.59 => 0.595. Sau 1 vài bước train, update được gọi
        # shadow = 0.5 * decay + (1- decay)*0.6
        # tiếp tục quá trình train với param=0.595 chứ không sử dụng giá trị của shadow để train
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=(1.0 - self.decay))
                # self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
           if name in self.shadow:
               self.backup[name] = param.data.clone()
               param.data = self.shadow[name].clone()
    
    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if hasattr(self, 'backup') and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}
        

def mixup_cutmix(images, targets, alpha=0.2, cutmix_prob=0.3):
    """ 
    + Return  (images, y_a, y_b, lam, is_cutmix)
    + mixup: pick two images randomly, mix them with ratio lam ~ Beta(alpha, alpha)
    + cutmix: pick two images randomly, cut a patch from one image and paste it to another image with ratio lam ~ Beta(alpha, alpha)
    + targets: one-hot encoded labels
    + using beta distribution to sample lam 
      => make sure lam is in (0, 1) => make new images remain more details in old images (with lam close to 0 or 1)
      => x = lam*x1 + (1-lam)*x2
    """
    device = images.device
    if np.random.rand() < cutmix_prob:
        # CutMix
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size, _, H, W = images.size()
        index = torch.randperm(batch_size ,device=device)
        cx, cy = np.random.randint(W), np.random.randint(H) # center of the box
        w = int(W * math.sqrt(1 - lam))
        h = int(H * math.sqrt(1 - lam))
        x0 = np.clip(cx - w // 2, 0, W)
        x1 = np.clip(cx + w // 2, 0, W)
        y0 = np.clip(cy - h // 2, 0, H)
        y1 = np.clip(cy + h // 2, 0, H)
        images[:, :, y0:y1, x0:x1] = images[index, :, y0:y1, x0:x1]
        lam = 1 - ((x1 - x0)*(y1 - y0) / (W * H)) # recalculate lam based on the area of the box
        return images, targets, targets[index], lam, True
    else:
        # MixUp
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=device)
        mixed = lam * images + (1 - lam)*images[index, ...]
        return mixed, targets, targets[index], lam, False
    
def mix_criterion(logits, y_a, y_b, lam, reduction='mean'):
    return lam * F.cross_entropy(logits, y_a, reduction=reduction) + (1 - lam) * F.cross_entropy(logits, y_b, reduction=reduction) 


# =================================================================================================
#  Albumentations (giữ màu/texture)
# =================================================================================================
class GrayWorldWB(A.ImageOnlyTransform):
    def __init__(self, p=0.35):
        super().__init__(p=p)
    
    def apply(self, img, **params):
        img = img.astype(np.float32)
        r, g, b = img[:, :, 0].mean()+1e-6, img[:, :, 1].mean()+1e-6, img[:, :, 2].mean()+1e-6
        gray = (b + g + r)/3.0
        img[:, : , 0] *= gray / r
        img[:, :, 1] *= gray / g
        img[:, :, 2] *= gray / b
        return np.clip(img, 0, 255).astype(np.uint8)
    

class SaturationClamp(A.ImageOnlyTransform):
    def __init__(self, max_sat=1.35, p=0.5):
        super().__init__(p=p)
        self.max_sat = max_sat
    
    def apply(self, img, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * self.max_sat, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def build_transforms(phase, img_size=224):
    if phase == 'train':
        return A.Compose([
            A.SmallestMaxSize(max_size=int(img_size*1.15), p=1.0),
            A.RandomCrop(img_size, img_size, p=1.0),
            GrayWorldWB(p=0.35),
            SaturationClamp(max_sat=1.35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.2),
            A.OneOf([A.GaussianBlur(blur_limit=3, p=1.0), A.MotionBlur(blur_limit=3, p=1.0)], p=0.12),
            A.GaussNoise(std_range=(5.0, 30.0), p=0.2),
            A.RandomShadow(shadow_roi=(0,0.6,1,1), num_shadows_limit=(1,1), shadow_dimension=4, p=0.10),
            A.CoarseDropout(num_holes_range=(1,1), hole_height_range=(img_size // 24, img_size // 12), 
                             hole_width_range=(img_size // 24, img_size // 12), fill=0, p=0.1),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),
        ])
    else:
        return A.Compose([ A.Resize(img_size, img_size),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),])



# =================================================================================================
# Dataset wrapper
# =================================================================================================
class AlbImageFolder(Dataset):
    """
        Args:
            root: Path to image folder 
            transform: Albumentations transform
            samples: List of (path, label) tuples (if create subset)
            classes: List of class names
    """
    def __init__(self, root=None, transform=None, samples=None, classes=None):
        if samples is not None:
            self.samples = samples
            self.classes = classes
        else:
            if root is None:
                raise ValueError("Must provide either root or samples")
            imagefolder = ImageFolder(root)
            self.samples = imagefolder.samples
            self.classes = imagefolder.classes

        self.transform = transform
        self.targets = [s[1] for s in self.samples]  # Extract labels
    

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        if self.transform: 
            img = self.transform(image=img)['image']
        return img, y

def subset_by_indices(base_ds, indices, transform):
    subset_samples = [base_ds.samples[i] for i in indices]
    
    return AlbImageFolder(
        samples=subset_samples,
        classes=base_ds.classes,
        transform=transform
    )
