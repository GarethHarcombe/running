import numpy as np
import math
import fitdecode
import copy


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


laps = []
points = []

with fitdecode.FitReader('/home/gareth/Downloads/Track.fit') as fit_file:
    lap = []
    for frame in fit_file:
        if isinstance(frame, fitdecode.records.FitDataMessage):
            if frame.name == 'lap':
                print(frame.get_value('total_elapsed_time'), frame.get_value('avg_speed'), frame.get_value('total_distance'), frame.get_value('start_time'))
                laps.append(copy.deepcopy(lap))
                lap = []
            
            elif frame.name == 'record':
                if frame.has_field('position_lat') and frame.has_field('position_long') and frame.has_field('timestamp') and frame.has_field('speed'):
                    point = Point(frame.get_value('timestamp'), frame.get_value('position_lat'), frame.get_value('position_long'), frame.get_value('speed'))
                    lap.append(point)
                    points.append(point)


lengths = [len(lap) for lap in laps]

print(lengths)


#gpx_file = open('/home/gareth/Downloads/Track.gpx', 'r')

#gpx = gpxpy.parse(gpx_file)

#points = []
#times = []

#for track in gpx.tracks:
    #for segment in track.segments:
        #for point in segment.points:
            #points.append((point.latitude, point.longitude))
            #times.append(point.time)

#raw_speeds = []

#for i in range(1, len(points)):
    #meters_sec_speed = haversine_dist(points[i], points[i - 1]) / (times[i] - times[i - 1]).seconds
    #min_km_speed = (100 / 6) / (meters_sec_speed + 1e-10)
    #raw_speeds.append(min_km_speed)
    
##print(raw_speeds)