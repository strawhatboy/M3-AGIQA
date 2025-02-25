
import torch
import torch.utils.data
import torch.nn as nn

from dataset import AGIDataset, AIGCIQA20kDataset

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)

    def get_dataset_class(self):
        if self.args['dataset_class'] == 'AGIDataset':
            return AGIDataset
        elif self.args['dataset_class'] == 'AIGCIQA-20k':
            return AIGCIQA20kDataset
        return AGIDataset
    
