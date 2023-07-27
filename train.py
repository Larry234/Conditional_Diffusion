import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image

from typing import Tuple
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import wandb
from accelerate import Accelerator

from models.embedding import *
from models.unet import UNet
from models.engine import ConditionalGaussianDiffusionTrainer, DDIMSampler
from datasets import CustomImageDataset
from utils import GradualWarmupScheduler



def main(args):
    
    # initialize W&B
    wandb.init(
        project=args.exp,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "dataset": args.data.split('/')[-1],
            "architecture": "classifier-free conditional DDPM",
            "num_res_blocks": args.num_res_blocks,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "T": args.num_timestep,
            "ch_mult": [1, 2, 2, 4]
        },
        job_type="training"
    )
    
    accelerator = Accelerator()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_ds = CustomImageDataset(root=args.data, transform=transform)
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # generate samples for each class in evaluation
    n_samples = args.num_condition[0] * args.num_condition[1]
    
    # define models
    model = UNet(
        T = args.num_timestep,
        num_labels = args.num_condition[0],
        num_atr = args.num_condition[1],
        ch = args.emb_size,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks = args.num_res_blocks,
        dropout=0.15,
    )
    
    trainer = ConditionalGaussianDiffusionTrainer(model, args.beta, args.num_timestep)
    
    # optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    sampler = DDIMSampler(
        model,
        beta=args.beta,
        T=args.num_timestep,
        w=args.w
    )
    
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=args.epochs, 
        eta_min=0, 
        last_epoch=-1
    )
    
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=2.5,
        warm_epoch=args.epochs // 10, 
        after_scheduler=cosineScheduler
    )
    
    
    dataloader, model, trainer, sampler, optimizer = accelerator.prepare(dataloader, model, trainer, sampler, optimizer)
    
    for epoch in range(1, args.epochs + 1):
        
        model.train()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        # train
        for x, c1, c2 in progress_bar:
            x = x.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            B = x.size()[0]
            
            loss = trainer(x, c1, c2).sum() / B ** 2.
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            
            # log training process
            wandb.log({"loss": loss.item()})
            
            progress_bar.set_postfix({
                "loss": f"{loss.item(): .4f}",
                "lr": optimizer.state_dict()['param_groups'][0]["lr"]
            })
            optimizer.step()
            optimizer.zero_grad()
            
        warmUpScheduler.step()
            
        # validation, save an image of currently generated samples
        size = x.size()[1:]
        if epoch % args.eval_interval == 0:
            model.eval()
            # sample random noise
            x_i = torch.randn(n_samples, *size).to(device)

            # create conditions of each class
            # create conditions like [0,0,0,1,1,1, ...] [0,1,2,3,0,1,2,3, ...]
            c1 = torch.arange(0, args.num_condition[0])
            c2 = torch.arange(0, args.num_condition[1])
            c1 = c1.repeat(n_samples // args.num_condition[0], 1).permute(1, 0).reshape(-1)
            c2 = c2.repeat(n_samples // args.num_condition[1])

            c1, c2 = c1.to(device), c2.to(device)

            x0 = sampler(x_i, c1, c2, steps=args.steps)
            
            # save image
            os.makedirs(os.path.join('result', args.exp), exist_ok=True)
            save_image(x0, os.path.join('result', args.exp, f'epoch_{epoch}.png'))
        
            # log image
            x0 = x0.permute(0, 2, 3, 1)
            x0 = x0.cpu().detach().numpy()
            
            images = [x0[i, :, :, :] for i in range(n_samples)]
            wandb.log({f"evalution epoch {epoch}": [wandb.Image(image) for image in images]})
            
            # save model
            save_root = os.path.join('checkpoints', args.exp)
            os.makedirs(save_root, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_root, f"model_{epoch}.pth"))
                
                





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, help='dataset location')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--img_size', type=int, default=128, help='training image size')
    parser.add_argument('--num_res_blocks', type=int, default=3, help='number of residual blocks in unet')
    parser.add_argument('--num_timestep', type=int, default=50, help='number of timesteps')
    parser.add_argument('--beta', type=Tuple[float, float], default=(0.0001, 0.02), help='beta start, beta end')
    parser.add_argument('--emb_size', type=int, default=10, help='embedding output dimension')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')
    parser.add_argument('--eval_interval', type=int, default=10, help='Frequency of evaluation')
    parser.add_argument('--exp', type=str, default='exp', help='experiment directory name')
    parser.add_argument('--sample_method', type=str, default='ddim', choices=['ddpm', 'ddim'], help='sampling method')
    parser.add_argument('--eta', type=float, default=0., help='ddim parameter when sampling')
    parser.add_argument('--steps', type=int, default=100, help='decreased timesteps using ddim')
    
    args = parser.parse_args()
    
    main(args)