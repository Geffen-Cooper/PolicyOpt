import pandas as pd
import os
import numpy as np
import re
from scipy.signal import resample
import argparse
from pathlib import Path

# ============ Argument parser ============
parser = argparse.ArgumentParser(description='Preprocess DSADS Dataset')
parser.add_argument('--root_dir', type=str, default="~/Projects/data/dsads", help='path to dsads dataset')
parser.add_argument('--activity_list', nargs='+', type=int, help='List of integers')
parser.add_argument('--body_part', type=str, default="right_leg", help='which body part to use')
parser.add_argument('--LOOCV', action='store_true', help='save data per user for LOOCV')


args = parser.parse_args()


# ============ global constants ============
body_parts = ['torso',
              'right_arm',
              'left_arm',
              'right_leg',
              'left_leg']

# active_body_parts = ['right_leg']
active_body_parts = [args.body_part]

NUM_USERS = 8

# training_users = [1,2,3,4,5]
# val_users = [6]
# testing_users = [7,8]

users = [1,2,3,4,5,6,7,8]
train_frac = 0.7
val_frac = 0.1
test_frac = 0.2

# data is 25 Hz sampling rate
og_sampling_rate = 25

window_len = 8
overlap_frac = 0.5

SEGMENT_LEN = 125 # samples per segment file, 5 seconds * 25 Hz 
NUM_SEGMENTS = 60 # 5 minutes (300 seconds) / 5 second segments

sensors = ['acc','gyro','mag']
sensor_dims = 3 # XYZ
channels_per_sensor = len(sensors)*sensor_dims

# dict to get index of sensor channel by bp and sensor
sensor_channel_map = {
    bp: 
    {
        sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
                          bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
                for sensor_i,sensor in enumerate(sensors)
    } for bp_i,bp in enumerate(body_parts)
}
active_channels = []
for bp in active_body_parts:
    for sensor in sensors:
        if sensor == 'acc':
            active_channels.append(sensor_channel_map[bp][sensor])
active_channels = np.array(active_channels).flatten()

num_samples_per_activity = SEGMENT_LEN*NUM_SEGMENTS

# ============ determine the labeling scheme ============
label_map = {
             0:'sitting',
             1:'standing',
             2:'lying on back',
             3:'lying on right side',
             4:'ascending stairs',
             5:'descending stairs',
             6:'standing in elevator',
             7:'moving in elevator',
             8:'walking in parking lot',
             9:'walking on flat treadmill',
             10:'walking on inclined treadmill',
             11:'running on treadmill,',
             12:'exercising on stepper',
             13:'exercising on cross trainer',
             14:'cycling on exercise bike horizontal',
             15:'cycling on exercise bike vertical',
             16:'rowing',
             17:'jumping',
             18:'playing basketball'
             }

active_label_map = {
    i:label_map[j] for i,j in enumerate(args.activity_list)
}

# ============ load remaining args ============
new_sampling_rate = og_sampling_rate
root_dir = os.path.expanduser(args.root_dir)


if __name__ == '__main__':
    # load the data
    activity_folders = os.listdir(root_dir)
    other_folders = []
    for folder in activity_folders:
        if len(folder) != 3:
            other_folders.append(folder)
    for folder in other_folders:
        activity_folders.remove(folder)
    activity_folders.sort(key=lambda f: int(re.sub('\D', '', f))) # dont include other preprocessed data
    participant_folders = os.listdir(os.path.join(root_dir,activity_folders[0]))
    participant_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
    segment_files = os.listdir(os.path.join(root_dir,activity_folders[0],participant_folders[0]))
    segment_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # load the data and labels

    activities = active_label_map.keys()

    if not args.LOOCV:
        train_pool_len = int(num_samples_per_activity*len(users)*train_frac)
        val_pool_len = int(num_samples_per_activity*len(users)*val_frac)
        test_pool_len = int(num_samples_per_activity*len(users)*test_frac)

        train_seg_len = int(num_samples_per_activity*train_frac)
        val_seg_len = int(num_samples_per_activity*val_frac)
        test_seg_len = int(num_samples_per_activity*test_frac)


        training_data_pool = {act: np.zeros((train_pool_len,len(active_channels))) for act in activities}
        val_data_pool = {act: np.zeros((val_pool_len,len(active_channels))) for act in activities}
        testing_data_pool = {act: np.zeros((test_pool_len,len(active_channels))) for act in activities}

        training_label_pool = {act: np.zeros(train_pool_len) for act in activities}
        val_label_pool = {act: np.zeros(val_pool_len) for act in activities}
        testing_label_pool = {act: np.zeros(test_pool_len) for act in activities}

        # merge data for each participant
        for user_i, user_folder in enumerate(participant_folders):
            for activity_i, activity_folder in enumerate(activity_folders):
                # print(f"user: {user_i}, activity: {label_map[activity_i]}")
                if activity_i in args.activity_list:
                    active_activity_i = args.activity_list.index(activity_i)
                else:
                    continue
                
                print(f"user: {user_i}, active_activity_i: {active_activity_i}, activity: {active_label_map[active_activity_i]}, pool_idx: {user_i*train_seg_len}, active_channels: {active_channels}")
                # create the data array which contains samples across all segment files
                data_array = np.zeros((num_samples_per_activity,len(active_channels)))
                label_array = np.zeros(num_samples_per_activity)
                for segment_i, segment_file in enumerate(segment_files):
                    data_file_path = os.path.join(root_dir,activity_folder,user_folder,segment_file)
                    data_segment = pd.read_csv(data_file_path,header=None).values
                    start = segment_i*SEGMENT_LEN
                    end = start + SEGMENT_LEN
                    data_array[start:end,:] = data_segment[:,active_channels]
                    label_array[start:end] = active_activity_i
                
                # add to pool
                # index the training seg, val seg, and test seg
                training_data_pool[active_activity_i][user_i*train_seg_len:user_i*train_seg_len+train_seg_len,:] = data_array[:train_seg_len,:]
                val_data_pool[active_activity_i][user_i*val_seg_len:user_i*val_seg_len+val_seg_len,:] = data_array[train_seg_len:train_seg_len+val_seg_len,:]
                testing_data_pool[active_activity_i][user_i*test_seg_len:user_i*test_seg_len+test_seg_len,:] = data_array[train_seg_len+val_seg_len:,:]
                
                training_label_pool[active_activity_i][user_i*train_seg_len:user_i*train_seg_len+train_seg_len] = label_array[:train_seg_len]
                val_label_pool[active_activity_i][user_i*val_seg_len:user_i*val_seg_len+val_seg_len] = label_array[train_seg_len:train_seg_len+val_seg_len]
                testing_label_pool[active_activity_i][user_i*test_seg_len:user_i*test_seg_len+test_seg_len] = label_array[train_seg_len+val_seg_len:]
        
        # merge training, validation, and test data each into arrays
        training_data = np.concatenate([training_data_pool[i] for i in training_data_pool.keys()])
        val_data = np.concatenate([val_data_pool[i] for i in val_data_pool.keys()])
        testing_data = np.concatenate([testing_data_pool[i] for i in testing_data_pool.keys()])

        training_labels = np.concatenate([training_label_pool[i] for i in training_label_pool.keys()])
        val_labels = np.concatenate([val_label_pool[i] for i in val_label_pool.keys()])
        testing_labels = np.concatenate([testing_label_pool[i] for i in testing_label_pool.keys()])


        # windowing on training/validation data to train classifier    
        slide = int(window_len*(1-overlap_frac))
        training_window_idxs = np.arange(0,training_data.shape[0]-window_len,slide)
        training_window_labels = training_labels[training_window_idxs]

        val_window_idxs = np.arange(0,val_data.shape[0]-window_len,slide)
        val_window_labels = val_labels[val_window_idxs]

        test_window_idxs = np.arange(0,testing_data.shape[0]-window_len,slide)
        test_window_labels = testing_labels[test_window_idxs]

        # print(val_data.shape[0],window_len,slide,np.arange(0,val_data.shape[0]-window_len,slide))

        # print("val_window_idxs:",val_window_idxs)
        # print("val_window_labels:",val_window_labels)
        # print("val_labels:",val_labels)


        print(f'Train shape: {training_data.shape},{training_labels.shape}')
        print(f'Val shape: {val_data.shape},{val_labels.shape}')
        print(f'Test shape: {testing_data.shape},{testing_labels.shape}')


        # ==================================================================


        folder = f"{root_dir}/merged_preprocess"
        Path(folder).mkdir(parents=True, exist_ok=True)
        np.save(f"{folder}/training_data",training_data)
        np.save(f"{folder}/training_labels",training_labels)

        np.save(f"{folder}/val_data",val_data)
        np.save(f"{folder}/val_labels",val_labels)

        np.save(f"{folder}/testing_data",testing_data)
        np.save(f"{folder}/testing_labels",testing_labels)
        np.save(f"activity_list",np.array(args.activity_list))


        # classifier data
        np.save(f"{folder}/training_window_idxs",training_window_idxs)
        np.save(f"{folder}/training_window_labels",training_window_labels)

        np.save(f"{folder}/val_window_idxs",val_window_idxs)
        np.save(f"{folder}/val_window_labels",val_window_labels)

        np.save(f"{folder}/testing_window_idxs",test_window_idxs)
        np.save(f"{folder}/testing_window_labels",test_window_labels)

    else:
        # we separate the data by participant
        training_data = {user: [] for user in range(NUM_USERS)} # raw data
        training_labels = {user: [] for user in range(NUM_USERS)} # raw labels
        training_window_idxs = {user: [] for user in range(NUM_USERS)} # window start and stop
        training_window_labels = {user: [] for user in range(NUM_USERS)} # label for each window
        window_partitions = {user: [] for user in range(NUM_USERS)} # 0 for train, 1 for validation

        # keep track of list indices as accumulate data from each file
        curr_train_window_idxs = {user: 0 for user in range(NUM_USERS)}


        # merge data for each participant into a numpy array
        for user_i, user_folder in enumerate(participant_folders):
            for activity_i, activity_folder in enumerate(activity_folders):
                active_activity_i = activity_i
                # print(f"user: {user_i}, activity: {label_map[activity_i]}")
                # if using an activity subset, need contiguous labels
                if activity_i in args.activity_list:
                    active_activity_i = args.activity_list.index(activity_i)
                else:
                    continue
                # print(f"user: {user_i}, active_activity_i: {active_activity_i}, activity: {self.activity_subset[active_activity_i]}, pool_idx: {user_i*train_seg_len}, self.active_channels: {self.active_channels}")
                # create the data array which contains samples across all segment files
                data_array = np.zeros((num_samples_per_activity,len(active_channels)))
                label_array = np.zeros(num_samples_per_activity)
                for segment_i, segment_file in enumerate(segment_files):
                    data_file_path = os.path.join(root_dir,activity_folder,user_folder,segment_file)
                    data_segment = pd.read_csv(data_file_path,header=None).values
                    start = segment_i*SEGMENT_LEN
                    end = start + SEGMENT_LEN
                    data_array[start:end,:] = data_segment[:,active_channels]
                    label_array[start:end] = active_activity_i

                # form windows
                slide = int(window_len*(1-overlap_frac))
                start_idxs = np.concatenate([np.array([curr_train_window_idxs[user_i]]),
                                            np.arange(curr_train_window_idxs[user_i]+slide,
                                            curr_train_window_idxs[user_i]+data_array.shape[0]-window_len,
                                            slide)]) # [0+offset,25+offset,50+offset,...,7450+offset]
                
                # split into training, validation
                num_train_samples = int(len(start_idxs)*(1-0.2))
            
                end_idxs = start_idxs + window_len
                train_window_labels = np.array([activity_i]*len(start_idxs))
                win_partitions = np.zeros(len(start_idxs))
                win_partitions[num_train_samples+1:] = 1 # validation windows for this activity, no overlap
                curr_train_window_idxs[user_i] += data_array.shape[0]

                # put into list
                training_data[user_i].append(data_array)
                training_labels[user_i].append(label_array)

                training_window_idxs[user_i].append(np.stack([start_idxs,end_idxs]).T)
                training_window_labels[user_i].append(train_window_labels)
                window_partitions[user_i].append(win_partitions)

        
        # now concatenate and save data
        folder = f"{args.root_dir}/LOOCV_preprocessed_data"
        os.mkdir(folder)
        for user_i in range(NUM_USERS):
            training_data[user_i] = np.concatenate(training_data[user_i])
            
            training_labels[user_i] = np.concatenate(training_labels[user_i])
            training_window_idxs[user_i] = np.concatenate(training_window_idxs[user_i])
            training_window_labels[user_i] = np.concatenate(training_window_labels[user_i])
            window_partitions[user_i] = np.concatenate(window_partitions[user_i])

            
            np.save(f"{folder}/data_{user_i}",training_data[user_i])
            np.save(f"{folder}/labels_{user_i}",training_labels[user_i])
            np.save(f"{folder}/window_idxs_{user_i}",training_window_idxs[user_i])
            np.save(f"{folder}/window_labels_{user_i}",training_window_labels[user_i])
            np.save(f"{folder}/window_partitions_{user_i}",window_partitions[user_i])