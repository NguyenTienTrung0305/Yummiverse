from utils.util import *
from utils.tta import *
import torch
import torch.nn as nn
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from tdqm import tdqm

class Trainer:
  def __init__(self, cfg):
    self.cfg = cfg
    set_seed(cfg.get('seed', 42))
    self.device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
  
  def train_cv(self, data_root):
    data_root = Path(data_root)
  