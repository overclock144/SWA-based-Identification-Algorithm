import torch
from torch.optim import lr_scheduler

from model_c import model

optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optim, step_size=3, gamma=0.3)
