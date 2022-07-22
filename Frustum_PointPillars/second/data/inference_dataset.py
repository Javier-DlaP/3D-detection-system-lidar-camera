import torch

class InferDataset(torch.utils.data.Dataset):
    def __init__(self, voxels_dicts):
        super(InferDataset, self,).__init__()

        self.voxels_dicts = voxels_dicts

    def __len__(self):
        return len(self.voxels_dicts)

    def __getitem__(self, idx):
        return self.voxels_dicts[idx]