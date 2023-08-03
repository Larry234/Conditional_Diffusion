import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset, TensorDataset

from glob import glob
import os

class PointDataset(Dataset):
    def __init__(self, root, transform=None, ignored=None, mean=2.5, std=2.5):
        
        self.root = root
        self.transform = transform
        self.points = np.array([])
        self.context = np.array([])
        self.mean = mean
        self.std = std
        
        for path in glob(os.path.join(self.root, '**', '*.npy')):
            class_label = path.split('/')[-2]
            if class_label == ignored:
                continue
            
            # load points in numpy format
            p = np.load(path)
            p = (p - mean) / std
            n = p.shape[0]
            
            conA, conB = class_label.split(' ')
            conA, conB = int(conA[-1]) - 1, int(conB[-1]) - 1
            cond = np.expand_dims(np.array([conA, conB]), axis=0)
            cond = cond.repeat(n, axis=0)
            
            self.points = np.concatenate((self.points, p), axis=0) if self.points.size else p
            self.context = np.concatenate((self.context, cond), axis=0) if self.context.size else cond
        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.to_list()
        
        points = torch.from_numpy(self.points[index].astype('float32'))
        context = torch.from_numpy(self.context[index].astype('int64'))
        
        return points, context
    

class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None, ignored=None):
        self.root = root
        self.transform = transform
        self.image_path = glob(os.path.join(root, '**', '*.jpg'))
        if ignored != None:
            self.image_path = [f for f in self.image_path if ignored not in f]
        self.obj_dict = {}
        self.atr_dict = {}
        obj = []
        atr = []
        labels = [label.split(' ') for label in os.listdir(root)]
        
        for l in labels:
            obj.append(l[0])
            atr.append(l[1])
        
        obj = list(set(obj))
        atr = list(set(atr))
         
        for i in range(len(obj)):
            self.obj_dict[obj[i]] = i
        
        for i in range(len(atr)):
            self.atr_dict[atr[i]] = i
            
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()
        image = Image.open(self.image_path[index]).convert('RGB')
        label = self.image_path[index].split('/')[-2]
        label = label.split(' ')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(self.obj_dict[label[0]], dtype=torch.int64), torch.tensor(self.atr_dict[label[1]], dtype=torch.int64)
    
    def get_class(self):
        return self.obj_dict, self.atr_dict
    
    
    
def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def generate_circle(center, radius, num_samples) -> np.ndarray:
    """ generate points inside a circle with cetner and radius """
    theta = np.linspace(0, 2*np.pi, num_samples)
    centerX, centerY = center
    a, b = radius * np.cos(theta) + centerX, radius * np.sin(theta) + centerY

    r = np.random.rand((num_samples)) * radius
    x, y = r * np.cos(theta) + centerX, r * np.sin(theta) + centerY
    return np.stack((x, y), axis=1)