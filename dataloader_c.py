import torch.utils.data

from dataset_c import train_data, test_data
from parm import batch_size

train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)
dataloader = {'train': train_loader, 'test': test_loader}
