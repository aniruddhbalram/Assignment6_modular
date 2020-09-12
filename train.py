from tqdm import tqdm
import torch.nn.functional as F
from cuda_avail import cuda_avail_device
device=cuda_avail_device
def train(model, device, train_loader, optimizer, epoch,regularizer,losses_array, acc_array):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Predict
        y_pred = model(data)
        # Calculate loss
        loss = F.nll_loss(y_pred,target)
        l1=0
        for p in model.parameters():      
            l1=l1+p.abs().sum()
        lambda_l1=1e-5
        # Backpropagation
        # For l1
        #------------------
        if(regularizer=='l1'):
            loss=loss+lambda_l1*l1       
        # For l2 loss
        #------------------
        # Nothing needed as this is accomodated in the optimizer
        # For l1 and l2 loss
        #------------------
        if(regularizer=='l1l2'):
            loss=loss+lambda_l1*l1      
        # For GBN
        #------------------
        # No need to do anything as GBN is addressed in model definition
        # For GBN + l1l2 
        #------------------
        if(regularizer=='l1l2'):
            loss=loss+lambda_l1*l1      
        losses_array.append(loss)
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly. 
        loss.backward()
        optimizer.step()
        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Model={regularizer} Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        acc_array.append(100*correct/processed)
