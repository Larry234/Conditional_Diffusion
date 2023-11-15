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
from models.engine import TripleCondDDIMSamplerEncoder, TripleConditionDiffusionEncoderTrainer
from dataset import CustomImageDatasetTripleCond
from utils import GradualWarmupScheduler, get_model, get_optimizer, get_piecewise_constant_schedule, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, LoadEncoder
from config import *



def main(args):
    
    # initialize W&B
    wandb.init(
        project=args.exp,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "dataset": args.data.split('/')[-1],
            "num_res_blocks": args.num_res_blocks,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "ch_mult": args.channel_mult,
            "guidance scale": args.w
        },
        job_type="training"
    )
    
    accelerator = Accelerator()
    
    CFG = TripleCond()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_ds = CustomImageDatasetTripleCond(root=args.data, transform=transform, ignored=args.ignored)
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # generate samples for each class in evaluation
    n_samples = args.num_condition[0] * args.num_condition[1] * args.num_condition[2]
    num_size = args.num_condition[0]
    num_atr = args.num_condition[1]
    num_obj = args.num_condition[2]
    args.num_condition[0] = len(CFG.SIZE2IDX)
    args.num_condition[1] = len(CFG.ATR2IDX)
    args.num_condition[2] = len(CFG.OBJ2IDX)
    # define models
    model = get_model(args)
    
    # optimizer and learning rate scheduler
    optimizer = get_optimizer(model, args)
    
    encoder = LoadEncoder(args)
    trainer = TripleConditionDiffusionEncoderTrainer(
        encoder = encoder,
        model = model,
        beta = args.beta,
        T = args.num_timestep,
    )

    sampler = TripleCondDDIMSamplerEncoder(
        model = model,
        encoder = encoder,
        beta = args.beta,
        T = args.num_timestep,
        w = args.w,
    )
    
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=args.epochs, 
        eta_min=0, 
        last_epoch=-1
    )
    
    if args.lr_schedule == "cosine":
        warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=2.5,
        warm_epoch=args.epochs // 10, 
        after_scheduler=cosineScheduler
        )
    elif args.lr_schedule == "piecewise":
        warmUpScheduler = get_piecewise_constant_schedule(optimizer, "1:30,0.1:60,0.05")
    
    elif args.lr_schedule == "linear":
        warmUpScheduler = get_linear_schedule_with_warmup(optimizer, args.epochs // 10, args.epochs)
    elif args.lr_schedule == "polynomial":
        warmUpScheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.epochss // 10, args.epochs)
    
    
    dataloader, model, trainer, sampler, optimizer = accelerator.prepare(dataloader, model, trainer, sampler, optimizer)
    
    for epoch in range(1, args.epochs + 1):
        
        model.train()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        # train
        for batch in progress_bar:
            x = batch["image"].to(device)
            c1 = [CFG.SIZE2IDX[s] for s in batch["size"]]
            c2 = [CFG.ATR2IDX[a] for a in batch["atr"]]
            c3 = [CFG.OBJ2IDX[o] for o in batch["obj"]]
            c1 = torch.tensor(c1, dtype=torch.long, device=device)
            c2 = torch.tensor(c2, dtype=torch.long, device=device)
            c3 = torch.tensor(c3, dtype=torch.long, device=device)
            B = x.shape[0]
            
            drop_c1 = torch.randn(c1.shape[0], device=c1.device) < args.drop_prob
            drop_c2 = torch.rand(c2.shape[0], device=c2.device) < args.drop_prob
            drop_c3 = torch.rand(c3.shape[0], device=c3.device) < args.drop_prob
            c1 = torch.where(drop_c1, args.num_condition[0], c1)
            c2 = torch.where(drop_c2, args.num_condition[1], c2)
            c3 = torch.where(drop_c3, args.num_condition[2], c3)
            
            loss = trainer(x, c1, c2, c3).sum() / B ** 2.
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            # log training process
            wandb.log({"loss": loss.item()})
            
            progress_bar.set_postfix({
                "loss": f"{loss.item(): .4f}",
                "lr": optimizer.state_dict()['param_groups'][0]["lr"]
            })
            optimizer.step()
            optimizer.zero_grad()
        
        wandb.log({'lr': optimizer.param_groups[0]["lr"]})
            
        warmUpScheduler.step()
            
        # validation, save an image of currently generated samples
        size = x.size()[1:]
        if epoch % args.eval_interval == 0:
            model.eval()
            # sample random noise
            x_i = torch.randn(n_samples, *size).to(device)

            # create conditions of each class
            # create conditions like [0,0,0,1,1,1, ...] [0,1,2,3,0,1,2,3, ...]
            c1 = torch.cat([torch.full((num_atr * num_obj,), i) for i in range(num_size)], dim=-1)
            c2 = torch.arange(0, num_atr)
            c3 = torch.arange(0, num_obj)
            c2 = c2.repeat(num_obj, 1).permute(1, 0).reshape(-1)
            c3 = c3.repeat(num_atr)
            c2 = c2.repeat(num_size)
            c3 = c3.repeat(num_size)
            c1, c2, c3 = c1.to(device), c2.to(device), c3.to(device)

            x0 = sampler(x_i, c1, c2, c3, steps=args.steps)
            
            # save image
            os.makedirs(os.path.join('result', args.exp, args.dir), exist_ok=True)
            save_image(x0, os.path.join('result', args.exp, args.dir, f'epoch_{epoch}.png'))
        
            # log image
            x0 = x0.permute(0, 2, 3, 1)
            x0 = x0.cpu().detach().numpy()
            c1, c2, c3 = c1.cpu().detach().numpy(), c2.cpu().detach().numpy(), c3.cpu().detach().numpy()
            images = [(f"{CFG.IDX2SIZE[c1[i]]} {CFG.IDX2ATR[c2[i]]} {CFG.IDX2OBJ[c3[i]]}", x0[i, :, :, :]) for i in range(n_samples)]
            wandb.log({f"evalution epoch {epoch}": [wandb.Image(image, caption=label) for label, image in images]})
            
            # save model
            save_root = os.path.join('checkpoints', args.exp, args.dir)
            os.makedirs(save_root, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_root, f"model_{epoch}.pth"))
                
                





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # General Hyperparameters 
    parser.add_argument('--arch', type=str, default='unetencoderattention', help='unet architecture')
    parser.add_argument('--data', type=str, default='/root/notebooks/nfs/work/dataset/toy_dataset_366_500', help='dataset location')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--eval_interval', type=int, default=10, help='Frequency of evaluation')
    parser.add_argument('--fix_emb', action='store_true', help='Freeze embedding table wieght to create fixed label embedding')
    parser.add_argument('--lr_schedule', type=str, default="cosine", choices=["cosine", "piecewise", "linear", "polynomial"])
    
    # ccip parameters
    parser.add_argument('--encoder_path', type=str, default=None, help="pretrained weight path of class encoder")
    parser.add_argument('--only_encoder', action="store_true", help="only use class encoder in ccip model(two tokens)")
    
    # Data hyperparameters
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=128, help='training image size')
    parser.add_argument('--dir', type=str, default="NoMiss")
    
    # Diffusion hyperparameters
    parser.add_argument('--num_timestep', type=int, default=1000, help='number of timesteps')
    parser.add_argument('--beta', type=Tuple[float, float], default=(0.0001, 0.02), help='beta start, beta end')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--eta', type=float, default=0., help='ddim parameter when sampling')
    parser.add_argument('--exp', type=str, default='exp', help='experiment directory name')
    parser.add_argument('--dir', type=str, default='NoMiss', help='model weight directory')
    parser.add_argument('--sample_method', type=str, default='ddim', choices=['ddpm', 'ddim'], help='sampling method')
    parser.add_argument('--steps', type=int, default=100, help='decreased timesteps using ddim')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='probability of dropping label when training diffusion model')
    
    # UNet hyperparameters
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of residual blocks in unet')
    parser.add_argument('--emb_size', type=int, default=128, help='embedding output dimension')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')
    parser.add_argument('--concat', action="store_true", help="concat label embedding before CA")
    parser.add_argument('--use_spatial_transformer', action="store_true", help="use transfomer based model to do attention")
    
    # Transformer hyperparameters(Optional)
    parser.add_argument('--projection_dim', type=int, default=512, help='q, k, v dimension in attention layer')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='attention head channels')
    parser.add_argument('--num_heads', type=int, default=-1, help='number of attention heads, either specify head_channels or num_heads')
    parser.add_argument('--channel_mult', type=list, default=[1, 2, 2, 2], help='width of unet model')
    parser.add_argument('--ignored', type=str, nargs='+', default=None, help='exclude folder when loading dataset, for compositional zero-shot generation')
    
    
    args = parser.parse_args()
    
    main(args)