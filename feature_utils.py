from torch.utils.data import Dataset

class DoubleFeatureDataset(Dataset):
    def __init__(self, err_stream_state, std_stream_state, device):
        self.err_stream_state = err_stream_state
        self.std_stream_state = std_stream_state
        self.device = device
        
    def __len__(self):
        return self.err_stream_state.size(0)
    
    def __getitem__(self, idx):
        return idx, self.err_stream_state[idx].to(self.device), self.std_stream_state[idx].to(self.device)
    
class FeatureDataset(Dataset):
    def __init__(self, inps, device):
        self.inps = inps
        self.device = device
        
    def __len__(self):
        return self.inps.size(0)
    
    def __getitem__(self, idx):
        return idx, self.inps[idx].to(self.device)