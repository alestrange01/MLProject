import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(333)
# Definire una classe custom per il dataset
class ChestXrayDataset(Dataset):
    def __init__(self, img_dir, transform=True):
        self.img_dir = img_dir

        if(transform):
            self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5), 
                            transforms.RandomVerticalFlip(0.5),
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5)
                            ])
        else: 
            self.transform = None

        
        self.img_list = os.listdir(self.img_dir)
        if('.DS_Store' in self.img_list): #per MAC
            self.img_list.remove('.DS_Store')

        #Dividi il dataset in train set e temp set (combina validation set e test set)
        train_temp_set, self.test_set = train_test_split(self, test_size=0.2, random_state=42)
        #Dividi il temp set in validation set e test set
        self.train_set, self.val_set = train_test_split(train_temp_set, test_size=0.25, random_state=42)
        
    def __len__(self):
        return len(self.img_list)
    
    def __getlabel__(self, img_name):
        if('NORMAL' in img_name.split('_')):
            return 0
        elif('PNEUMONIA' in img_name.split('_')):
            return 1
        else: return 2

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        label = self.__getlabel__(img_name)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_loaders(self, batch_size, shuffle, drop_last, num_workers):
        return {"train": DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers),
           "val": DataLoader(self.val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
           "test": DataLoader(self.test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)} 
    
    def get_trainset(self):
        return self.train_set
    
    def get_valset(self):
        return self.val_set
    
    def get_testset(self):
        return self.test_set
    


