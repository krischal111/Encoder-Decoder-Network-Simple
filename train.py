import mynet_model
import torch
import torch.nn as nn
from torch.optim import Adam

from dotmap import DotMap
import os.path
import time
import importlib
import datasets
importlib.reload(datasets)
from datasets import *

device = torch.device("mps")

# Instantiate models
model = None
model_file = "mynet_mnist.pt"

# Optimizer and Loss Function
optimizer = None
loss_fn = None

# Training History
H = None
history_file = "training_history.pt"

def instantiate_models():
    print("Instantiating Models")
    global model, optimizer, loss_fn, H

    # Instantiate Model
    if os.path.isfile(model_file):
        model = torch.load(model_file).to(device)
    else:
        model = mynet_model.myNet().to(device)
    
    optimizer = Adam(model.parameters(), 0.01)
    loss_fn = nn.MSELoss()

    # Training History
    if os.path.isfile(history_file):
        H = torch.load(history_file)
    else:
        H = DotMap({
            "train_loss":[],
            "validation_loss":[],
        })
    
    prepare_loaders()
    i = len(dataset.trainLoader)
    j = dataset.batch_size
    return model

def save_models():
    torch.save(model, model_file)
    torch.save(H, history_file)

def train_one_epoch(debug=False):
    if debug:
        print("Training one epoch")

    start =  time.time()
    model.train()

    total_train_loss = 0
    total_validation_loss = 0

    # Train a batch
    for (x, y) in dataset.trainLoader:
        x = x.to(device)
        
        reconstruction = model(x)
        loss = loss_fn(reconstruction, x)
        total_train_loss += loss.cpu().detach().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_time = time.time()
    training_time = train_time - start
    if debug:
        print("Training took ", training_time)
    
    # Validation step
    with torch.no_grad():
        model.eval()
        for (x,y) in dataset.valLoader:
            x = x.to(device)
            
            recon = model(x)
            loss = loss_fn(recon, x)
            total_validation_loss += loss.cpu().detach().numpy()
    
    avg_train_loss = total_train_loss / dataset.trainSteps
    avg_validation_loss = total_validation_loss / dataset.valSteps

    # Add to history
    H.train_loss.append(avg_train_loss)#.cpu().detach().numpy())
    H.validation_loss.append(avg_validation_loss) # .cpu().detach().numpy())

    end = time.time()
    total_time = end - start

    summary = DotMap({
        "Train Loss": total_train_loss,
        "Validation Loss": total_validation_loss,
        "Training time" : training_time,
        "Time Taken": total_time,
    })
    return summary