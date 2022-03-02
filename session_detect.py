import numpy as np
import torch
import math
import fitdecode
from gps_loader import GPSDataset
from torch.utils.data import DataLoader


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


def main():
    gps_dataset = GPSDataset("/home/gareth/Documents/Programming/running/export_12264640/activities.csv",
                         "/home/gareth/Documents/Programming/running/export_12264640/")    
    train_dataloader = DataLoader(gps_dataset, batch_size=64, shuffle=True)
    
    
    


if __name__ == "__main__":
    main()