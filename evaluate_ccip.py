import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import argparse

import os
from glob import glob

from utils import LoadEncoder
from config import *

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default="/root/notebooks/nfs/work/dataset/toy_dataset_66_500", help="path of test dataset")
    parser.add_argument('--encoder_path', type=str, default="checkpoints/CCIP/model_10.pth")
    parser.add_argument('--num_condition', type=int, nargs='+', help="number of classes in each condition")
    parser.add_argument('--target', type=str, nargs='+', help="evaluation target class in dataset")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LoadEncoder(args).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    num_atr = len(IDX2ATR)
    num_obj = len(IDX2OBJ)
    total = num_atr * num_obj
    
    atr = torch.arange(0, num_atr)
    obj = torch.arange(0, num_obj)
    atr = atr.repeat(total // num_atr, 1).permute(1, 0).reshape(-1)
    obj = obj.repeat(total // num_obj)
    atr, obj = atr.to(device), obj.to(device)
    
    # get class embedding
    with torch.no_grad():
        class_features = model.class_encoder(atr=atr, obj=obj)
        class_features = model.class_projection(class_features)
    
    # zero-shot classification
    for label in args.target:
        images = glob(os.path.join(args.data, label, "*.jpg"))
        images = [transform(Image.open(image).convert("RGB")) for image in images]
        images = torch.stack(images).to(device)
        labels = torch.full((len(images),), CLS2IDX[label], device=device)
        
        with torch.no_grad():
            image_features = model.image_encoder(images)
            image_features = model.image_projection(image_features)
            
            image_features = F.normalize(image_features, p=2, dim=-1)
            class_features = F.normalize(class_features, p=2, dim=-1)
            dot_similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
            
        top1, top5 = accuracy(dot_similarity, labels, (1, 5))
        print(f"\nEvaluation class \"{label}\":\n    Top1 accuracy: {top1.item():.2f} %\n    Top5 accuracy: {top5.item():.2f} %")
    
    
    
    
    