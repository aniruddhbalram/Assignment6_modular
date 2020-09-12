from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Transforms:
    def __init__(self,mean,std_dev):
        self.mean = mean
        self.std_dev = std_dev
    def train_transforms(self):
        train_transforms_ = transforms.Compose([
                                            #  transforms.Resize((28, 28)),
                                            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                            transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((self.mean,), (self.std_dev,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                            # Note the difference between (0.1307) and (0.1307,)
                                            ])
        return train_transforms_
    def test_transforms(self):
        test_transforms_ = transforms.Compose([
                                            #  transforms.Resize((28, 28)),
                                            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize((self.mean,), (self.std_dev,))
                                            ])
        return test_transforms_
