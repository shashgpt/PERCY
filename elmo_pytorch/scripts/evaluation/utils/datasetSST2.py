from config import *

class DatasetSST2(Dataset):

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        sentence = self.x[index]
        label = self.y[index]
        return sentence, label