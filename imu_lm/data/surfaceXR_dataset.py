import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple, List

from shared.features import FeatureExtractor
from shared.constants import SURFACE_COLS, HEAD_COLS, HAND_COLS, IMU_COLS, HAND_INDICES, IMU_WINDOW_SIZE
from shared.config import USE_SURFACE, USE_DISTANCE
from shared.butterworth import ButterFilters


# Butterworth filter for IMU data
butter_filters = ButterFilters(fs=200)

# load normalization parameters
with open('norm_params.pkl', 'rb') as f:
    norm_params = pickle.load(f)

imu_min = norm_params['imu_min'].reshape(1, -1)
imu_max = norm_params['imu_max'].reshape(1, -1)
epsilon = 1e-7  # to avoid division by zero


class JointDataset(Dataset):
    def __init__(self, data_list: List[Dict]):
        self.feature_extractor = FeatureExtractor()
        self.windows = []
        self.touch_labels = []
        self.gesture_labels = []
        self.surfaces = []
        
        # aggregate data from all users
        for i, data_dict in enumerate(data_list):
            # extract features for each window
            processed_windows = [
                self.feature_extractor.extract_features(window, surface=data_dict['surface'] if USE_SURFACE else None, use_distance=USE_DISTANCE)
                for window in data_dict['hand_windows']
            ]
            
            # reshape processed windows
            processed_windows = np.array(processed_windows)
            if processed_windows.ndim > 3:
                N, T, J, C = processed_windows.shape                        # [N, 60, 34, 3]
                processed_windows = processed_windows.reshape(N, T, J * C)  # [N, 60, 102]
            
            processed_windows = processed_windows[:, :, HAND_INDICES]

            self.windows.extend(processed_windows)
            self.touch_labels.extend(data_dict['touch_labels'])
            self.gesture_labels.extend(data_dict['gesture_labels'])
            self.surfaces.extend([data_dict['surface']] * len(data_dict['hand_windows']))
        
        # convert to tensors
        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.touch_labels = torch.tensor(np.array(self.touch_labels), dtype=torch.long)
        self.gesture_labels = torch.tensor(np.array(self.gesture_labels), dtype=torch.long)
        self.surfaces = torch.tensor(np.array(self.surfaces), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'window': self.windows[idx],
            'touch_label': self.touch_labels[idx],
            'gesture_label': self.gesture_labels[idx],
            'surface': self.surfaces[idx]
        }


class MultimodalDataset(Dataset):
    def __init__(self, data_list: List[Dict], include_gesture: bool=True):
        self.feature_extractor = FeatureExtractor()
        self.imu_windows = []
        self.hand_windows = []
        self.touch_labels = []
        self.gesture_labels = []
        self.surfaces = []
        self.include_gesture = include_gesture

        # aggregate data from all users
        for data_dict in data_list:
            # extract features for each window
            processed_windows = [
                self.feature_extractor.extract_features(window, surface=data_dict['surface'] if USE_SURFACE else None, use_distance=USE_DISTANCE)
                for window in data_dict['hand_windows']
            ]

            # reshape processed windows
            processed_windows = np.array(processed_windows)
            if processed_windows.ndim > 3:
                N, T, J, C = processed_windows.shape                        # [N, 60, 34, 3]
                processed_windows = processed_windows.reshape(N, T, J * C)  # [N, 60, 102]
            
            processed_windows = processed_windows[:, :, HAND_INDICES]
            
            self.hand_windows.extend(processed_windows)
            self.imu_windows.extend(data_dict['imu_windows'])  # [N, 200, 24]
            self.surfaces.extend([data_dict['surface']] * len(data_dict['hand_windows']))
            self.touch_labels.extend(data_dict['touch_labels'])
            if include_gesture:
                self.gesture_labels.extend(data_dict['gesture_labels'])
        
        # convert to tensors
        self.hand_windows = torch.tensor(np.array(self.hand_windows), dtype=torch.float32)
        self.imu_windows = torch.tensor(np.array(self.imu_windows), dtype=torch.float32)
        self.touch_labels = torch.tensor(np.array(self.touch_labels), dtype=torch.long)
        if include_gesture:
            self.gesture_labels = torch.tensor(np.array(self.gesture_labels), dtype=torch.long)
        self.surfaces = torch.tensor(np.array(self.surfaces), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.hand_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'hand_window': self.hand_windows[idx],
            'imu_window': self.imu_windows[idx],
            'touch_label': self.touch_labels[idx],
            'gesture_label': self.gesture_labels[idx] if self.include_gesture else torch.zeros(1),
            'surface': self.surfaces[idx]
        } 
    

def create_lopo_datasets(dataset: Dict, test_user: str, dataset_type: Dataset=MultimodalDataset, include_gesture: bool=True) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train, validation, and test datasets using LOPO method."""
    # list of users excluding test user
    train_users = [user for user in dataset.keys() if user != test_user]
    
    # aggregate data for training and test/val
    train_data = []
    for user in train_users:
        train_data.extend(dataset[user])
        
    test_data = dataset[test_user]
    
    # create datasets
    train_dataset = dataset_type(train_data, include_gesture=include_gesture)
    valid_dataset = dataset_type(test_data, include_gesture=include_gesture)  # using test user's data for validation
    test_dataset = dataset_type(test_data, include_gesture=include_gesture)   # using test user's data for testing
    
    print(f"\nLOPO Split Info:")
    print(f"Test/Validation User: {test_user}")
    print(f"Training Users: {train_users}")
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Valid/Test: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset, test_dataset
    

def create_training_datasets(dataset: Dict, splits: Tuple[float, float]=(0.8, 0.1), dataset_type: Dataset=MultimodalDataset) -> Tuple[Dataset, Dataset, Dataset]:
    # aggregate all user data
    all_data = []
    for user_data in dataset.values():
        all_data.extend(user_data)
    
    # create single dataset
    print("Processing dataset...")
    full_dataset = dataset_type(all_data)
    
    # split dataset
    train_split, valid_split = splits
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    valid_size = int(valid_split * total_size)
    test_size = total_size - train_size - valid_size
    
    return torch.utils.data.random_split(
        full_dataset, 
        [train_size, valid_size, test_size]
    )


DELAY_THRESHOLD = 60

def normalize_timestamps(df):
    """Normalize timestamps for both Unity and IMU data."""
    unity_start = df['UnixTimeQuest'].iloc[0]
    watch_start = df['AccTimeStamp0'].iloc[0]
    
    df = df.copy()
    df['unity_norm_time'] = df['UnixTimeQuest'] - unity_start
    df['watch_norm_time'] = df['AccTimeStamp0'] - watch_start

    time_diff = abs(df['unity_norm_time'] - df['watch_norm_time'])
    df['VALID'] = time_diff < DELAY_THRESHOLD    
    
    return df, unity_start, watch_start


def get_hand_data(df, cols):
    """Extract unique hand tracking frames by dropping duplicates."""
    # drop repeated hand tracking values to get 60Hz data
    hand_df = df.drop_duplicates(subset=df.columns[16:118]).reset_index(drop=True)
    return hand_df, hand_df[cols]


def get_imu_data(df, hand_window, window_size_imu, imu_cols):
    """Find synchronized IMU data for a given hand tracking window."""
    # get the timestamp of the last sample in this hand tracking window
    end_time = hand_window['unity_norm_time'].iloc[-1]
    
    # find the matching index dataset
    match_idx = (df['unity_norm_time'] - end_time).abs().idxmin()
    
    # extract IMU window (counting backward from the matching point)
    imu_start_idx = max(0, match_idx - window_size_imu + 1)
    imu_window = df.iloc[imu_start_idx:match_idx+1]
    
    if len(imu_window) < window_size_imu:
        print(f"match_idx: {match_idx}")
        print(f"imu_start_idx: {imu_start_idx}")
        print(f"IMU window has insufficient samples. Skipping...", len(imu_window))
        return None
    
    if not all(imu_window['VALID']):
        return None
    
    imu_windows = torch.tensor(imu_window[imu_cols].values) # [200, 6]
    imu_windows = 2 * ((imu_windows - imu_min) / (imu_max - imu_min + epsilon)) - 1  # normalize to [-1, 1]
    imu_features = butter_filters.forward(imu_windows.T) # [6, 4, 200]
    imu_features = imu_features.permute(2, 0, 1).reshape(IMU_WINDOW_SIZE, -1).numpy()

    return imu_features


def create_training_windows(df: pd.DataFrame, window_size_hand: int=60, window_size_imu: int=200, stride: int=20):
    """Create sliding windows of synchronized hand tracking and IMU data for training."""
    # reset dataframe indices
    df = df.reset_index(drop=True)

    # normalize timestamps
    df, _, _ = normalize_timestamps(df)
    
    # extract hand tracking data
    hand_df, _ = get_hand_data(df, HEAD_COLS+HAND_COLS)
    
    # surface features remain constant (if specified)
    surface = np.array(df[SURFACE_COLS].value_counts().index[0])
    
    # store windows and labels
    windows_hand, windows_imu, labels = [], [], []

    for i in range(0, len(hand_df) - window_size_hand, stride):
        # get window of hand tracking data
        hand_window = hand_df.iloc[i:i+window_size_hand]
        
        if len(hand_window) < window_size_hand:
            continue  # skip incomplete windows

        if not all(hand_window['VALID']):
            continue  # skip windows with invalid timestamps        
        
        # get synchronized IMU data
        imu_features = get_imu_data(df, hand_window, window_size_imu, IMU_COLS) # [200, 6]
        
        if imu_features is None:
            continue  # skip if we don't have enough IMU samples
        
        # extract hand features and labels
        hand_features = hand_window[HEAD_COLS+HAND_COLS].values 
        label = hand_window['isTouched1'].values
        
        windows_hand.append(hand_features)
        windows_imu.append(imu_features)
        labels.append(label)

    print(f"Created {len(windows_hand)} synchronized windows.")
    
    user_data = {
        'hand_windows': np.array(windows_hand),
        'imu_windows': np.array(windows_imu),
        'labels': np.array(labels),
        'surface': surface
    }

    return user_data


def create_dataset(data_path: str, processed_data_path: str='/data/projects/SurfaceXR/dataset.pkl'):
    """Create dataset from all user data."""
    dataset = {}

    # check if the processed dataset file exists
    if os.path.exists(processed_data_path):
        print(f"Loading processed dataset from {processed_data_path}...")
        with open(processed_data_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    # build dataset from raw data
    with open(data_path, 'rb') as f: 
        data = pickle.load(f)
        for user in data:
            print(f"Processing: {user}")
            print("-"*20)
            user_data = []
            for surface_type in data[user]: # 'WALL' and 'TABLE'
                # create training data windows from path tracing data
                path_pos_data = data[user][surface_type]['PATH']['POSITIVE']
                if len(path_pos_data) > 0:
                    user_data.append(create_training_windows(path_pos_data, stride=20))

                # create training data window from negative path tracing data
                path_neg_data = data[user][surface_type]['PATH']['NEGATIVE']
                if len(path_neg_data) > 0:  
                    user_data.append(create_training_windows(path_neg_data, stride=40))

            total_windows = sum(len(data_dict['hand_windows']) for data_dict in user_data)
            print(f"Total Windows: {total_windows}")
            print("-"*20)
            dataset[user] = user_data

        with open(processed_data_path, 'wb') as f:
            pickle.dump(dataset, f)        
            
    return dataset