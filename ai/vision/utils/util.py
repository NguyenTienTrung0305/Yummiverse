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
    # Ví dụ trong quá trình train (param=0.6, shadow=0.5) => param.data = 0.6. Update được gọi
        # shadow = 0.5 * decay + (1- decay)*0.6 = 0.52
        # tiếp tục quá trình train với param=0.6 chứ không sử dụng giá trị của shadow để train => param.data=0.8 => shadow=0.52*decay+(1-decay)*0.8
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


    def state_dict(self):
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']
        

def mixup_cutmix(images, targets, alpha=0.2, cutmix_prob=0.3):
    """ 
    + Return  (images, y_a, y_b, lam, is_cutmix)
    + mixup: pick two images randomly, mix them with ratio lam ~ Beta(alpha, alpha)
    + cutmix: pick two images randomly, cut a patch from one image and paste it to another image with ratio lam ~ Beta(alpha, alpha)

    + images: (batch_size, 3, H, W)
    + targets: one-hot encoded labels (batch_size,)
    + using beta distribution to sample lam 
      => make sure lam is in (0, 1) => make new images remain more details in old images (with lam close to 0 or 1)
      => x = lam*x1 + (1-lam)*x2
    """
    device = images.device
    if np.random.rand() < cutmix_prob:
        # CutMix
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size, _, H, W = images.size()
        index = torch.randperm(batch_size, device=device)   # generate random permutations from 1 to batchsize
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
#  Albumentations (giữ màu/texture) for training
# =================================================================================================
class GrayWorldWB(A.ImageOnlyTransform):
    def __init__(self, alpha_min=0.1, alpha_max=0.3, p=0.3):
        super().__init__(p)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
    
    def apply(self, img, alpha=0.2, **params):
        img = img.astype(np.float32)
        r, g, b = img[:, :, 0].mean()+1e-6, img[:, :, 1].mean()+1e-6, img[:, :, 2].mean()+1e-6
        gray = (b + g + r)/3.0
        
        corr = img.copy()
        corr[..., 0] *= gray / r
        corr[..., 1] *= gray / g
        corr[..., 2] *= gray / b
        
        # alpha blend
        out = (1 - alpha) * img + alpha * corr
        return np.clip(out, 0, 255).astype(np.uint8)
    
    def get_params(self):
        return {"alpha": float(np.random.uniform(self.alpha_min, self.alpha_max))}
    

class SaturationClamp(A.ImageOnlyTransform):
    def __init__(self, sat_min=0.85, sat_max=1.25, p=0.5):
        super().__init__(p)
        self.sat_min = sat_min
        self.sat_max = sat_max
    
    def apply(self, img, s=1.0, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def get_params(self):
        return {"s": float(np.random.uniform(self.sat_min, self.sat_max))}


def build_det_transforms(phase, img_size=224):
    if phase == 'train':
        return A.Compose([
            # remaining aspect ratio + padding + random crop => avoid distortion
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT101, p=0.35),
            
            GrayWorldWB(alpha_min=0.1, alpha_max=0.25, p=0.25),
            SaturationClamp(sat_min=0.90, sat_max=1.15, p=0.4),
            

            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.2),
            A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.2),
            A.OneOf([A.GaussianBlur(blur_limit=(3,5), p=1.0), A.MotionBlur(blur_limit=5, p=1.0)], p=0.12),
            A.GaussNoise(std_range=(5.0, 30.0), p=0.2),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),
        ])
    else:
        return A.Compose([
                A.LongestMaxSize(max_size=img_size, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
                A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),
            ])

def build_cls_transforms(phase, img_size=224):
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size),scale=(0.7, 1.0),ratio=(0.85, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(0.03, 0.08, 8, border_mode=cv2.BORDER_REFLECT_101, p=0.35),
            
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.RandomGamma((80,120)),
                A.CLAHE((1.0,2.0), tile_grid_size=(8,8)),
            ], p=0.6),
            

            A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=6, val_shift_limit=6, p=0.2),
            A.RandomBrightnessContrast(0.18, 0.18, p=0.5),
            A.OneOf([A.MotionBlur(5), A.GaussianBlur((3,5))], p=0.15),
            A.OneOf([A.GaussNoise((5.0,25.0)), A.ISONoise((0.01,0.03),(0.1,0.3))], p=0.2),
            A.OneOf([
                A.Downscale(scale_range=(0.6, 0.9),  
                            interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_CUBIC}
                            ), 
                A.ImageCompression(quality_range=(65, 95))], p=0.2),

            A.CoarseDropout(num_holes_range=(1,1), hole_height_range=(img_size // 28, img_size // 14), 
                             hole_width_range=(img_size // 28, img_size // 14), fill=0, p=0.1),
            A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),
        ])
    else:
        return A.Compose([ 
                A.Resize(img_size, img_size),
                A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()
            ])


# =================================================================================================
#  preprocess image for inferences
# =================================================================================================
def preprocess_image_segment(img: np.ndarray):
    """Apply mild denoise + contrast normalization for real camera images"""
    if img.dtype != np.uint8:
        img = (255 * np.clip(img,0,1)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
    img = cv2.GaussianBlur(img, (3,3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def morphology_clean(mask: np.ndarray):
    """Clean mask artifacts (small holes, noise)"""
    mask = mask.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)

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
            self.classes = imagefolder.classes       # List of unique class names

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


