from utils.util import *
from utils.tta import *
import torch
import torch.nn as nn
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from models.freshness_classifier import *

from tqdm import tqdm


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
  

  def _eval_metrics(self, model, loader, report=False, class_names=None):
    model.eval()
    all_predict, all_y = [], []
    for img, y in loader:
      img = img.to(self.device)
      logits = model(img)
      predict = logits.argmax(1).cpu().numpy()
      all_predict.extend(predict)
      all_y.extend(y.numpy())

    all_predict, all_y = np.array(all_predict), np.array(all_y)
    acc = (all_predict == all_y).mean()
    f1score = f1_score(all_y, all_predict, average='macro')
    if report:
      print(classification_report(all_y, all_predict, target_names=class_names, digits=4))
      print("Confusion matrix:\n", confusion_matrix(all_y, all_predict))
    return acc, f1score




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

      # Optim: discriminative LR (head lr higher)
      head_params = list(model.head.parameters())
      backbone_params = [param for name, param in model.named_parameters() if 'head' not in name]
      optimizer = torch.optim.AdamW([
        {
          'params': backbone_params,
          'lr': self.cfg['optimizer']['freshness'].get('lr_backbone', 1e-4)
        },
        {
          'params': head_params,
          'lr': self.cfg['optimizer']['freshness'].get('lr_head', 1e-3)
        }
      ], weight_decay=self.cfg['optimizer']['freshness'].get('weight_decay', 1e-4))

      # OneCycleLR (with warmup)
      steps_per_epoch = max(1, len(tr_loader))
      scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
          self.cfg['optimizer']['freshness'].get('lr_backbone', 1e-4) * 5.0,
          self.cfg['optimizer']['freshness'].get('lr_head', 1e-3) * 5.0
        ],
        epochs=self.cfg['training']['freshness'].get('epochs', 100),
        steps_per_epoch=steps_per_epoch,
        pct_start=self.cfg['optimizer']['freshness'].get('warmup_ratio', 0.1),
        anneal_strategy='cos',
        div_factor=self.cfg['optimizer']['freshness'].get('div_factor', 25.0),
      )


      # Loss 
      w = compute_class_weights([labels[i] for i in tr_idx], n_cls).to(self.device)
      ce = nn.CrossEntropyLoss(
        weight=w,
        label_smoothing=self.cfg['training']['freshness'].get('label_smoothing', 0.1)
      )


      # EMA
      ema = EMA(model=model, decay=self.cfg['training']['freshness'].get('ema_decay', 0.999))
      scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['training']['freshness'].get('amp', True))


      best_f1 = 0.0
      patience = 0
      for epoch in range(1, self.cfg['training']['freshness'].get('epochs', 100) + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{self.cfg['training']['freshness'].get('epochs', 100)} - Train")

        for images, targets in pbar:
          images = images.to(self.device)    # (batch_size, 3, H, W), pbar: wrapper of iterator tr_loader
          targets = targets.to(self.device)  # (batch_size)

          # Mixup + Cutmix
          use_mix = (epoch <= int(self.cfg['training']['freshness'].get('epochs', 100) 
                                  * self.cfg['training']['freshness'].get('mix_stop_ratio', 0.7))) and self.cfg['training']['freshness'].get('use_mix', True)
          if use_mix:
            images, y_a, y_b, lam, _ = mixup_cutmix(images, targets, 
                                                    alpha=self.cfg['training']['freshness'].get('mix_alpha', 0.2), 
                                                    cutmix_prob=self.cfg['training']['freshness'].get('mix_prob', 0.35))
            
          optimizer.zero_grad(set_to_none=True)
          with torch.cuda.amp.autocast(enabled=self.cfg['training']['freshness'].get('amp', True)):
            logits = model(images)
            loss = mix_criterion(logits=logits, y_a=y_a, y_b=y_b, lam=lam) if use_mix else ce(logits, targets)


          # Nhân hệ số loss với hệ số scale để tránh loss quá nhỏ khi dùng amp (FP16), không tính toán được gradient
          # Sau đó tính gradient tương ứng với loss đã bị scale
          scaler.scale(loss).backward() 

          # unscale gradient => đưa gradient về đúng mức ban đầu
          scaler.unscale_(optimizer) 

          # confined sum normalize of gradients to avoid exploding gradient => sudden weight change => model instability
          # if gradient too large => all gradient will be divide by total_norm / max_norm
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


          # overlook inf or NaN gradient
          # call optimizer.step() to update weights
          scaler.step(optimizer) 

          # update scale for next iteration
          scaler.update()

          # update learning rate scheduler 
          scheduler.step()

          # update weights for inference
          ema.update()

          tr_loss += loss.item()
          if not use_mix:
            preds = logits.argmax(1)
            tr_total += targets.size(0) 
            tr_correct += (preds == targets).sum().item()

          pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*tr_correct/max(1,tr_total):.2f}%")


        # Validation (EMA weights)
        ema.apply_to(model=model)
        val_acc, val_f1 = self._eval_metrics(model, val_loader, report=True, class_names=base_dataset.classes)
        ema.restore(model=model)




