from __future__ import print_function
import torch
from transforms import Transforms
cuda = torch.cuda.is_available()
class Dataloaders_:
    def __init__(self,shuffle_bool,batch_size,num_workers,pin_memory_bool,shuffle_bool_else,batch_size_else,train_,test_):
        self.shuffle_bool=shuffle_bool
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.pin_memory_bool=pin_memory_bool
        self.shuffle_bool=shuffle_bool_else
        self.batch_size_else=batch_size_else
        self.train=train_
        self.test=test_
        # dataloader arguments - something you'll fetch these from cmdprmt
        self.dataloader_args = dict(shuffle=self.shuffle_bool, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory_bool) if cuda else dict(shuffle=shuffle_bool_else, batch_size=self.batch_size_else)
    
    def train_loader_(self):
        # train dataloader
        train_loader = torch.utils.data.DataLoader(self.train, **self.dataloader_args)
        return train_loader

    def test_loader_(self):
        # test dataloader
        test_loader = torch.utils.data.DataLoader(self.test, **self.dataloader_args)
        return test_loader
