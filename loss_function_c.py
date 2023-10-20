import torch

from parm import dev

loss_fn = torch.nn.CrossEntropyLoss().to(dev)