import numpy as np

class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle = False, drop_last = False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()
        
        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx]}

class NewCriticDataset:
    def __init__(self, batch_size, obs, old_pred_values, target_values, shuffle = False, drop_last = False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.old_pred_values = old_pred_values.view(-1)
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()
        
        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.old_pred_values = self.old_pred_values[index]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :], 'old_pred_values': self.old_pred_values[start_idx:end_idx], 'target_values': self.target_values[start_idx:end_idx]}
