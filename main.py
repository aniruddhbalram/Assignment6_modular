import torch
from torch import optim
from torchsummary import summary
from model import Net
from model import NetGBN
from model import BatchNorm
from model import GhostBatchNorm
from dataloaders import Dataloaders_
from test import test
from train import train
from cuda_avail import cuda_avail_device 
from graph_analysis import plot_model_comparison
import matplotlib.pyplot as plt
import numpy as np
from transforms import Transforms
from torchvision import datasets, transforms
from tqdm import tqdm

EPOCHS=2
SEED=1
device=cuda_avail_device()
regularizers =['l1','l2','l1l2','gbn','gbnl1l2']
losses={}
accuracies={}
test_losses = {}
test_accuracies= {}
print(torch.cuda.is_available())
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
transforms_=Transforms(0.1307,0.3801)
train_ = datasets.MNIST('./data', train=True, download=True, transform=transforms_.train_transforms())
test_ = datasets.MNIST('./data', train=False, download=True, transform=transforms_.test_transforms())

dataloaders=Dataloaders_(True,128,0,True,True,64,train_,test_)
train_loader=dataloaders.train_loader_()
test_loader=dataloaders.test_loader_()

model = Net().to(device)
summary(model, input_size=(1, 28, 28))
model2 = NetGBN().to(device)
summary(model2, input_size=(1, 28, 28))

for regularizer in regularizers:
  model=None
  optimizer=None
  scheduler = None
  if(regularizer=='l1'):
    model= Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=7,gamma=0.2)
  if(regularizer=='l2'):
    model= Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=7,gamma=0.2)
  if(regularizer=='l1l2'):
    model= Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=7,gamma=0.2)
  if(regularizer=='gbn'):
    model = NetGBN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=7,gamma=0.2)
  if(regularizer=='gbnl1l2'):
    model=NetGBN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=7,gamma=0.2)
  losses[regularizer]=[]
  test_losses[regularizer]=[]
  accuracies[regularizer]=[]
  test_accuracies[regularizer]=[]
  for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch,regularizer,losses[regularizer],accuracies[regularizer])
    test(model, device, test_loader, test_losses[regularizer], test_accuracies[regularizer],regularizer)
    scheduler.step()

model1_acc_hist = test_accuracies['l1']
model2_acc_hist = test_accuracies['l2']
model3_acc_hist = test_accuracies['l1l2']
model4_acc_hist = test_accuracies['gbn']
model5_acc_hist = test_accuracies['gbnl1l2']
model1_loss_hist = test_losses['l1']
model2_loss_hist = test_losses['l2']
model3_loss_hist = test_losses['l1l2']
model4_loss_hist = test_losses['gbn']
model5_loss_hist = test_losses['gbnl1l2']

legend_list = ["l1", "l2", "L1+ L2", "GBN","GBN + L1L2"]
plot_model_comparison(legend_list,
                      model1_acc_hist, model1_loss_hist,
                      model2_acc_hist, model2_loss_hist,
                      model3_acc_hist, model3_loss_hist,
                      model4_acc_hist, model4_loss_hist,
                      model5_acc_hist, model5_loss_hist)






