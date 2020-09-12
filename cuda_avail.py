import torch
cuda = torch.cuda.is_available()
def cuda_avail_device():
    device = torch.device("cuda" if cuda else "cpu")
    return device
