from utils.util import *
from utils.tta import *
import torch
import torch.nn as nn
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from models.freshness_classifier import *

from tdqm import tdqm





class ClassifierTrainer:
  def __init__(self, cfg):
    self.cfg = cfg
    set_seed(cfg.get('training', {}).get('seed', 42))
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  def _build_loaders_classifier(self, base, tr_idx, val_idx):
    tr_transforms = build_transforms('train', self.cfg['training']['freshness'].get('img_size', 224))
    val_transforms = build_transforms('val', self.cfg['training']['freshness'].get('img_size', 224))
    tr_ds = subset_by_indices(base, tr_idx, tr_transforms)
    val_ds = subset_by_indices(base, val_idx, val_transforms)

    tr_loader = DataLoader(
      tr_ds, 
      batch_size=self.cfg['training']['freshness'].get('batch_size', 64), 
      shuffle=True, 
      num_workers=self.cfg['training'].get('workers', 4)
    )
    val_loader = DataLoader(
      val_ds, 
      batch_size=self.cfg['training']['freshness'].get('batch_size', 64), 
      shuffle=False, 
      num_workers=self.cfg['training'].get('workers', 4)
    )

    return tr_loader, val_loader

  """
    base_dataset: + contain all path to images and name classes (ex: ('data/cat/cat1.jpg', 0))
                  + base_dataset.targets => label index for each image
                  + base_dataset.classes => class names that the model can predict
  """
  def train_cv(self, data_root):
    data_root = Path(data_root)
    base_dataset = AlbImageFolder(str(data_root), transform=None)
    n_cls = len(base_dataset.classes)
    labels = base_dataset.targets
    skf = StratifiedKFold(n_splits=self.cfg['training'].get('folds', 5), shuffle=True, random_state=42)
    out_dir = Path(self.cfg.get('paths', {}).get('classifier_out_dir', 'runs/classifier'))
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.arange(len(base_dataset)), labels), 1):

      print(f"\n=== Fold {fold}/{self.cfg['training'].get('folds', 5)} ===")

      tr_loader, val_loader = self._build_loaders_classifier(base_dataset, tr_idx, val_idx)
      model = FreshnessClassifier(
        self.cfg['freshness_model'].get('name'), 
        n_cls, 
        self.cfg['training']['freshness'].get('dropout', 0.3), 
        pretrained=True
      ).to(self.device)
