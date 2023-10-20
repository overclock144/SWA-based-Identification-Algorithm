import torch
import torchvision.models
from torchvision import models

from parm import class_num, dev

model = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, class_num)
model.to(dev)