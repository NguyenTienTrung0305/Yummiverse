import torch
import torch.nn as nn
import timm

class FreshnessClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b2", num_classes=3, pretrained=True, dropout=0.3, global_pool="avg"):
        super(FreshnessClassifier, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool=global_pool)
        feat = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout*0.6),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
