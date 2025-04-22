import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
from sklearn.model_selection import KFold

gait_labels = {
    "dog_trot": 0,
    "dog_gallop": 1,
    "dog_pace": 2,
    "horse_trot": 3,
    "horse_walk": 4,
    "mpc_trot": 5,
    "mpc_gallop": 6,
    "mpc_bound": 7,
    "mpc_crawl": 8,
    "solo8_trot": 9,
    "solo8_bound": 10,
    "solo8_walk": 11,
    "solo8_crawl": 12,
    "solo8_stilt": 13,
    "solo8_wave": 14
}
class LocoDataset(Dataset):
    def __init__(self, dataset_path, split='train', split_ratio=0.8):
        assert split in ['train', 'val', 'full'], "please check the spell on variable split in class LocoDataset"
        assert 0 < split_ratio < 1, "split ratio should be between 0 and 1"
        
        self.split = split
        self.split_ratio = split_ratio
        self.dataset_path = dataset_path
        assert os.path.exists(self.dataset_path), f"{self.dataset_path} does not exist"
        
        assert self.dataset_path.endswith('.npz'), f"{self.dataset_path} is not an npz file"
        
        self.dataset_ = np.load(self.dataset_path, allow_pickle=True)
        self.source2label = dict(self.dataset_['source2label'].item())
        self.gait2label = dict(self.dataset_['gait2label'].item())
        self.label2source = {v: k for k, v in self.source2label.items()}
        self.label2gait = {v: k for k, v in self.gait2label.items()}
        self.full_data = self.dataset_['data']
        
        
        self.total_samples = int(self.full_data[-1, -1])
        
        self.info_dim = 3
        self.GAIT_INDEX = -3
        self.SOURCE_INDEX = -2
        
        
        self.epsilon = 1e-7
        

        self.n_actions = len(self.gait2label)
        self.n_sources = len(self.source2label)
        self.block_size = 100
        
        #### send the velocity ###

        self.joint_vel()
        self.dim = self.full_data.shape[1] - self.info_dim
  
        ##########################
        self.find_min_max()
        self.scale_data()
        self.seperate_data()
        
        if self.split in ['train', 'val']:
            self.split_data()

    def __len__(self):
        if self.split == 'train':
            return int(self.total_samples * self.split_ratio)
        elif self.split == 'val':
            return int(self.total_samples * (1 - self.split_ratio))
        elif self.split == 'full':
            return self.total_samples
        else:
            raise ValueError("Invalid split type")

    def __getitem__(self, idx):
        sample_ = (self.train_traj if self.split == 'train' else
                   self.val_traj if self.split == 'val' else
                   self.traj)[idx]
        
        tmp_data = sample_[:self.block_size, :-self.info_dim].copy()
        gait_label = sample_[0, self.GAIT_INDEX]
        source_label = sample_[0, self.SOURCE_INDEX]
        len_data = tmp_data.shape[0]
        
        if len_data < self.block_size:
            valid = np.zeros((self.block_size, 1))
            pad = np.zeros((self.block_size - len_data, self.dim))
            motion_data = np.concatenate((tmp_data, pad), axis=0)
            valid[:len_data] = 1
        else:
            motion_data = tmp_data[:self.block_size].copy()
            valid = np.ones((self.block_size, 1))

        unique_gait_label = gait_labels[self.label2source[source_label] + '_' + self.label2gait[gait_label]]
        return {
            'data': torch.tensor(motion_data, dtype=torch.float32),
            'motion_type': torch.tensor(gait_label, dtype=torch.int64),
            'motion_source': torch.tensor(source_label, dtype=torch.int64),
            'valid': torch.tensor(valid, dtype=torch.int64),
            'duration': torch.tensor(len_data, dtype=torch.int64),
            "gait_label": torch.tensor(unique_gait_label, dtype=torch.int64),
        }

    def scale_data(self):
        tmp = self.full_data[:, :-self.info_dim].copy()
        tmp = ((tmp - self.min_val) / (self.max_val - self.min_val + self.epsilon)) * 2 - 1
        self.scaled_data = np.hstack((tmp, self.full_data[:, -self.info_dim:]))
        # 

    def joint_vel(self):
        tmp = np.zeros_like(self.full_data[:, :-self.info_dim])
        tmp_full_data = self.full_data.copy()
        for i in range(1,self.full_data.shape[0]):
            tmp[i, -12:] = (self.full_data[i, -(12+self.info_dim):-self.info_dim] - self.full_data[i - 1, -(12 + self.info_dim): - self.info_dim]) / 0.02
        tmp_full_data = np.hstack((tmp_full_data[:,:-self.info_dim], tmp[:,-12:]))
        self.full_data = np.hstack((tmp_full_data, self.full_data[:, -self.info_dim:]))


    def find_min_max(self):
        self.min_val = np.min(self.full_data[:, :-self.info_dim], axis=0)
        self.max_val = np.max(self.full_data[:, :-self.info_dim], axis=0)


    def seperate_data(self):
        self.sample_index = np.unique(self.scaled_data[:, -1])
        self.traj= [self.scaled_data[self.scaled_data[:, -1] == i] for i in self.sample_index]

 
 
    def split_data(self):
        total_samples = len(self.traj)
        train_len = int(total_samples * self.split_ratio)
        inds = np.random.permutation(total_samples)
        self.train_traj = [self.traj[i] for i in inds[:train_len]]
        self.val_traj = [self.traj[i] for i in inds[train_len:]]

def get_data_loaders(split, batch_size=32, shuffle=True, num_workers=4):
    assert split in ['train', 'val','full'], "please check the spell on variable split in function get_data_loaders"
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'motion_retarget', 'go2_retarget', 'dataset.npz')
    assert os.path.exists(dataset_dir), f"{dataset_dir} does not exist"
    return DataLoader(LocoDataset(dataset_dir, split), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_K_data_loaders(batch_size=32):
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'motion_retarget', 'go2_retarget', 'dataset.npz')
    assert os.path.exists(dataset_dir), f"{dataset_dir} does not exist"
    full_dataset = LocoDataset(dataset_dir, split='full')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_train_loaders = []
    k_val_loaders = []
    
    for train_index, val_index in kf.split(full_dataset):
        trainloader = DataLoader(full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index), num_workers=4)
        testloader = DataLoader(full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_index), num_workers=4)
        k_train_loaders.append(trainloader)
        k_val_loaders.append(testloader)
    
    return k_train_loaders, k_val_loaders

