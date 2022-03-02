from torch.utils.data import Dataset, random_split
import torch
import os
import fitdecode
import gzip
import pandas as pd


class Point:
    def __init__(self, timestamp, lat, long, speed):
        self.timestamp = timestamp
        self.lat = lat
        self.long = long
        self.speed = speed
        
    def __repr__(self):
        return f'Point at lat: {self.lat}, long: {self.long}, at time {self.timestamp}, m/s: {self.speed} \n'


FILE = 0
RUN_TYPE = 1
RUN_TYPES = {"long_run": [1, 0, 0], "workout": [0, 1, 0], "other": [0, 0, 1]}
NUM_POINTS = 60 * 60 * 6


class GPSDataset(Dataset):
    """GPS dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gps_frame = pd.read_csv(csv_file)
        self.gps_frame = self.gps_frame[self.gps_frame["run_type"].notna()]
        self.gps_frame = self.gps_frame[self.gps_frame.Filename != '']
        
        for _, row in self.gps_frame.iterrows():
            with gzip.open(os.path.join(root_dir, row["Filename"]), 'rb') as f:
                file_content = f.read()
                with open(os.path.join(root_dir, row["Filename"][:-3]), 'wb') as w:
                    w.write(file_content)
        
        train_size = int(0.8 * len(self.gps_frame))
        test_size = len(self.gps_frame) - train_size
        self.train_dataset, self.test_dataset = random_split(self.gps_frame, [train_size, test_size])        
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        gps_path = os.path.join(self.root_dir, self.train_dataset.iloc[idx]["Filename"])[:-3]
        
        points = []
        with fitdecode.FitReader(gps_path) as fit_file:
            for frame in fit_file:
                if isinstance(frame, fitdecode.records.FitDataMessage):
                    if frame.name == 'lap':
                        #print(frame.get_value('total_elapsed_time'), frame.get_value('avg_speed'), frame.get_value('total_distance'), frame.get_value('start_time'))
                        pass
                        
                    elif frame.name == 'record':
                        if frame.has_field('position_lat') and frame.has_field('position_long') and frame.has_field('timestamp') and frame.has_field('speed'):
                            point = Point(frame.get_value('timestamp'), frame.get_value('position_lat'), frame.get_value('position_long'), frame.get_value('speed'))
                            points.append(point)        
        
        lats = [points[0].lat - point.lat for point in points] 
        lats = lats + (NUM_POINTS - len(lats)) * [0]
        
        longs = [points[0].long - point.long for point in points]
        longs = longs + (NUM_POINTS - len(longs)) * [0]
        
        timestamps = [(point.timestamp - points[0].timestamp).seconds for point in points]
        timestamps = timestamps + (NUM_POINTS - len(timestamps)) * [0]
        
        speeds = [point.speed for point in points]
        speeds = speeds + (NUM_POINTS - len(speeds)) * [0]
        
        run_type = self.train_dataset.iloc[idx]["run_type"]
        
        sample = {'data': torch.tensor([lats, longs, speeds, timestamps]), 'type': RUN_TYPES[run_type]}
        return sample
        
        
def main():
    dataset = GPSDataset("/home/gareth/Documents/Programming/running/export_12264640/activities.csv",
                         "/home/gareth/Documents/Programming/running/export_12264640/")
    
    print(dataset[0])  
        
        
if __name__ == "__main__":
    main()