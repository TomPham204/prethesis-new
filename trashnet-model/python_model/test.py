import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from PIL import Image

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, torchvision.models as models

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import helper

root=str(os.path.dirname(os.path.abspath(__file__)))

#Define model architecture & load the weight
count=0
categories={0: 'cardboard', 1: 'fabric', 2: 'glass', 3: 'metal', 4: 'paper', 5: 'plastic'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True, progress=True)
for i in model.children():
  count+=1
  if(count < 15):
    for param in i:
      param.requires_grad=False
model.classifier[1] = nn.Sequential(nn.Linear(1280, 6))
model_addon=nn.Sequential()
model=nn.Sequential(model_addon, model)
model.to(device)
state=torch.load(root+'\\trashfinal1006.pth', map_location=device)
model.load_state_dict(state, strict=False)
model.eval()


def predict(dir=root+'\\data\\dataset'):
   global root
   count=0
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   #Image transform
   image_transforms = {
       "test": transforms.Compose([
           transforms.Resize((250, 250)),
           transforms.ToTensor()
       ])
   }

   #Test dataset
   trash_dataset_og = datasets.ImageFolder(root = dir,transform = image_transforms["test"])
   idx2class = {v: k for k, v in trash_dataset_og.class_to_idx.items()}
   _, __, test_sampler = helper.create_samplers(trash_dataset_og, 0, 0)
   test_loader = DataLoader(dataset=trash_dataset_og, shuffle=False, batch_size=1, sampler = test_sampler)

   #Define model architecture
   model = models.mobilenet_v2(pretrained=True, progress=True)
   for i in model.children():
     count+=1
     if(count < 15):
       for param in i:
         param.requires_grad=False
   model.classifier[1] = nn.Sequential(nn.Linear(1280, len(idx2class)))
   model_addon=nn.Sequential()
   model=nn.Sequential(model_addon, model)
   model.to(device)

   #Load the weight
   state=torch.load(root+'\\trashfinal.pth', map_location=device)
   model.load_state_dict(state, strict=False)

   #Switch to eval mode from train mode
   model.eval()

   #Load the image and predict
   test_loader_iter = iter(test_loader)
   single_img, single_lbl = next(test_loader_iter)
   single_img, single_lbl = single_img.to(device), single_lbl.to(device)
   pred = torch.log_softmax(model(single_img), dim = 1)
   _, pred_class = torch.max(pred, dim = 1)
   pred_class = pred_class.item()
   single_img = single_img.squeeze().permute(1, 2, 0).cpu().numpy()
   print(f"True Class = {idx2class[single_lbl.item()]}")
   print(f"Pred Class = {idx2class[pred_class]}")

   return idx2class[pred_class], idx2class[single_lbl.item()], single_img

def predictSingleUnseen(imgdir=root+'\\data\\unseen\\demo4.jpg'):
   #Test dataset
   image_transforms =  transforms.Compose([transforms.Resize((250, 250)),transforms.ToTensor()])
   image = image_transforms(Image.open(imgdir)).unsqueeze(0).to(device)

   #Load the image and predict
   pred = torch.log_softmax(model(image), dim = 1)
   _, pred_class = torch.max(pred, dim = 1)
   
   pred_class = pred_class.item()
   image = image.squeeze().permute(1, 2, 0).cpu().numpy()
   return categories[pred_class],'', image

predictSingleUnseen()