import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ===========================================================================================
# TTA
# ===========================================================================================

# Test Time Augmentation => Applying image transformations in the inference stage
# When predicting, the model predicts the transformed images below one by one and then evaluates the result using the average of the results.
def build_tta(img_size=224):
  return [
    A.Compose([
      A.Resize(img_size, img_size), 
      A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), 
      ToTensorV2()
    ]),

    A.Compose([
      A.Resize(img_size, img_size), 
      A.HorizontalFlip(p=1.0), 
      A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), 
      ToTensorV2()
    ]),

    A.Compose([
      A.SmallestMaxSize(256), 
      A.CenterCrop(img_size,img_size), 
      A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), 
      ToTensorV2()
    ]),

  ]

def predict_tta(model, pil_img, device, tta_list):
  model.eval()
  img = np.array(pil_img.convert('RGB'))
  probs = []

  # apply each augmentation
  for tf in tta_list:
    x = tf(image=img)['image'].unsqueeze(0).to(device)
    p = torch.softmax(model(x), dim=1)
    probs.append(p)

  # average probability from each augmentation
  prob = torch.stack(probs).mean(0) #(1, num_classes)

  # get class have max probability
  conf, pred = prob.max(1)
  return int(pred.item()), float(conf.item()), prob.squeeze(0).cpu().numpy()
