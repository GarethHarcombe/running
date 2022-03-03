import numpy as np
import torch
import math
from gps_loader import get_dataloaders
from torch.utils.data import DataLoader
from torch import nn
from model import NeuralNetwork
from tqdm import tqdm


def haversine_dist(a, b):
    # Outputs distance in meters
    R = 6371 * 10 ** 3
    theta1 = a[0] * np.pi / 180
    theta2 = b[0] * np.pi / 180
    diff_theta = (b[0] - a[0]) * np.pi / 180
    diff_lambda = (b[1] - a[1]) * np.pi / 180
    
    a = np.sin(diff_theta / 2) * np.sin(diff_theta / 2) + np.cos(theta1) * np.cos(theta2) * np.sin(diff_lambda/2) * np.sin(diff_lambda/2)
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1-a))
    
    d = R * c
    return d


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for sample in tqdm(iter(dataloader)):
        # Compute prediction and loss
        pred = model(sample["data"])
        loss = loss_fn(pred, sample["type"])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 100 == 0:
            #loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for sample in tqdm(iter(dataloader)):
            pred = model(sample["data"])
            test_loss += loss_fn(pred, sample["type"]).item()
            correct += (pred == sample["type"]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    model = NeuralNetwork()
    
    dataloaders = get_dataloaders("/home/gareth/Documents/Programming/running/export_12264640/", "activities.csv")
    train_dataloader = dataloaders["Train"]
    test_dataloader = dataloaders["Test"]
    
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")    
    


if __name__ == "__main__":
    main()