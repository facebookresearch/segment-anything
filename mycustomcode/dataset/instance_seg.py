from torch.utils.data import Dataset


class BaseInstanceDataset(Dataset):
    def __init__(self):
        assert False, print("Unimplemented Dataset.")

    def __getitem__(self, item):
        pass
