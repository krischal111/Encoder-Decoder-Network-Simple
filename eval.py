import importlib
import train
importlib.reload(train)
import matplotlib.pyplot as plt
import torch
device = torch.device("mps")

f = plt.figure(figsize=(30,15))

def evaluate(model, savefig = False):
    for i in range(25):
        x, y = train.random_test_sample()
        pred = model(x.to(device).unsqueeze(1))

        plt.subplot(5,15, 3*i+1)
        plt.imshow(x[0], cmap="Blues")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Original")

        plt.subplot(5,15, 3*i+2)
        plt.imshow(pred[0,0].cpu().detach().numpy(), cmap="Blues")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Reconstructed")
    
    if savefig:
        plt.savefig(savefig)


    


    