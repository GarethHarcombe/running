from torch.utils.data import Dataset
import torch


class Point:
    def __init__(self, timestamp, lat, long, speed):
        self.timestamp = timestamp
        self.lat = lat
        self.long = long
        self.speed = speed
        
    def __repr__(self):
        return f'Point at lat: {self.lat}, long: {self.long}, at time {self.timestamp}, m/s: {self.speed} \n'


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
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 1  # len(self.landmarks_frame)

    def __getitem__(self, idx):
        points = []
        with fitdecode.FitReader('/home/gareth/Downloads/Track.fit') as fit_file:
            for frame in fit_file:
                if isinstance(frame, fitdecode.records.FitDataMessage):
                    if frame.name == 'lap':
                        print(frame.get_value('total_elapsed_time'), frame.get_value('avg_speed'), frame.get_value('total_distance'), frame.get_value('start_time'))
                        
                    elif frame.name == 'record':
                        if frame.has_field('position_lat') and frame.has_field('position_long') and frame.has_field('timestamp') and frame.has_field('speed'):
                            point = Point(frame.get_value('timestamp'), frame.get_value('position_lat'), frame.get_value('position_long'), frame.get_value('speed'))
                            points.append(point)        
        
        lats = [point.lat for point in points]
        longs = [point.long for point in points]
        timestamps = [point.timestamp for point in points]
        return torch.tensor([lats, longs, timestamps])
        
        
        #if torch.is_tensor(idx):
            #idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
                                #self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        #if self.transform:
            #sample = self.transform(sample)

        #return sample