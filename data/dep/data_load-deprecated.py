from . import data_sort
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


class data_load():     #isolate from data sort
    
    def __init__(self, do_sort=False, Num_classes=4,directory = "Dataset"):
        '''
        param : The directory name that contains the classes of files
        
        '''
        #sorted_data=data_sort()
        if not do_sort==False:
            self.Num_classes,self.dataset_dir=data_sort.data_sort().do_prep()
        else:
            self.Num_classes=Num_classes
            cwd = os.getcwd()
            self.dataset_dir = os.path.join(cwd, directory) 

    def target_to_oh(self, target):
        one_hot = torch.eye(self.Num_classes)[target]
        return one_hot
    
   
    def load_dataset(self, mode, batch_size):
        data_path = os.path.join(self.dataset_dir, mode)
        
        self.chosen_transforms = {
            'train': transforms.Compose
            ([
                transforms.Resize(size=256),
                #transforms.RandomRotation(degrees=10),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                #transforms.Normalize(self.mean_nums, self.std_nums),   USE IF COMMAND HERE FOR NORMALIZATION
                transforms.ToTensor()     
            ]), 
            'val': transforms.Compose
            ([
                transforms.Resize(256),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                #transforms.Normalize(self.mean_nums, self.std_nums),                
                transforms.ToTensor()
            ]),
        }

        dataset = datasets.ImageFolder(
            root=data_path,
            transform= self.chosen_transforms[mode], target_transform=self.target_to_oh
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,                                
            shuffle=True
        )
        return  data_loader, len(dataset), dataset.classes

    def do_prep(self,batch_size = 100):
        self.dataloaders = {}
        self.dataset_sizes = {}                 #100
        self.dataloaders['train'], self.dataset_sizes['train'], self.class_names = self.load_dataset('train', batch_size)
        self.dataloaders['val'], self.dataset_sizes['val'],_ = self.load_dataset('val', batch_size)
        return self.dataloaders, self.dataset_sizes, self.class_names
        

        

