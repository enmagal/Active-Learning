import torch
from PIL import Image

class PoolCustomDataset:
    def __init__(self, df, transform=None):
        nbrImg = len(df)
        df = df.reset_index(drop=True) 
        self.data = df
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = str(self.data.iloc[idx][0])
        image = Image.open(image_path)
        entropy = self.data.iloc[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, entropy
    
    def checkChannel(self, df):
        datasetRGB = []
        for index in range(len(df)):
            image_path = data_base + str(df.iloc[index][0])
            if (Image.open(image_path).getbands() == ("R", "G", "B")): # Check Channels
                datasetRGB.append(self.data.iloc[index])
        return datasetRGB

class TrainCustomDataset:
    def __init__(self, df, transform=None):
        nbrImg = len(df)
        df = df.reset_index(drop=True) 
        self.data = df
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = str(self.data.iloc[idx][0])
        image = Image.open(image_path)
        label = self.data.iloc[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def checkChannel(self, df):
        datasetRGB = []
        for index in range(len(df)):
            image_path = data_base + str(df.iloc[index][0])
            if (Image.open(image_path).getbands() == ("R", "G", "B")): # Check Channels
                datasetRGB.append(self.data.iloc[index])
        return datasetRGB