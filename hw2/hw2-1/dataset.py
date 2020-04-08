import h5py, json

from collections import namedtuple
from torch.utils.data import Dataset, DataLoader

class VCTK_Dataset(Dataset):
    def __init__(self, h5_path, index_path, mode='train', seg_len=128):
        self.dataset = h5py.File(h5_path, 'r')
        with open(index_path) as f_index:
            self.indexes = json.load(f_index)
        self.indexer = namedtuple('index', ['speaker', 'i', 't'])
        self.seg_len = seg_len
        self.mode = mode #train,test

    def __getitem__(self, i):
        index = self.indexes[i]
        index = self.indexer(**index)
        speaker = index.speaker
        i, t = index.i, index.t
        seg_len = self.seg_len
        data = [speaker, self.dataset[f'{self.mode}/{i}/'][t:t+seg_len]]
        return tuple(data)

    def __len__(self):
        return len(self.indexes)
    
    