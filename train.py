import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torch.nn.paraller import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size

import argparse
import matplotlib.pyplot as plt
import numpy as np

from models.embedding import *
from models.mlp import *
from models.ddpm_scheduler import NoiseScheduler
from utils.optimization import get_scheduler
from ddpm import *
from datasets import PointDataset




def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.ToTensor()
    train_ds = PointDataset(args.data, mean=2.5, std=2.5)
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # generate samples for each class in evaluation
    n_samples = args.num_condition[0] * args.num_condition[1]
    
    # define models
    
    # Embedding layers
    atr_embedding = LabelEmbedding(num_classes=5, hidden_size=args.emb_size, dropout_prob=0.1).to(device)
    obj_embedding = LabelEmbedding(num_classes=5, hidden_size=args.emb_size, dropout_prob=0.1).to(device)
    time_embedding = SinusoidalEmbedding(size=args.emb_size, device=device)
    
    # Noise predictor
    nn_model = MLP(
        hidden_layers=args.hidden_layer,
        input_size=2,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
    )
    
    # diffusion model
    ddpm = DDPM(
        nn_model=nn_model,
        betas=(1e-4, 0.02),
        n_T=args.num_timestep,
        device=device,
        drop_prob=0.1
    )
    
    # Noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=args.num_timestep,
        beta_schedule=args.beta_schedule
    )
    
    
    # optimizer and learning rate scheduler
    params = list(atr_embedding.parameters()) + list(obj_embedding.parameters()) + list(nn_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
    )
    
    frames = []
    losses = []
    
    for epoch in range(1, args.epochs + 1):
        
        nn_model.train()
        obj_embedding.train()
        atr_embedding.train()
        progress_bar = tqdm(dataloader, desc='Epoch')
        
        # train
        for x, c in progress_bar:
            x = x.to(device)
            c = c.to(device)
            B = x.size()[0]
            
            # sample noise from N(0 ,1)
            noise = torch.randn(x.size())
            
            # sample timesteps from U(0, num_timesteps)
            timesteps = torch.randint(0, args.num_timestep, (B,))
            
            x_t = noise_scheduler.add_noise(x.cpu(), noise, timesteps)
            
            # embed conditions and timesteps
            c1, c2 = c[:, 0], c[:, 1]
            c1 = obj_embedding(c1)
            c2 = atr_embedding(c2)
            t = time_embedding(timesteps)
            
            # predict noise with timestep and conditions
            noise_pred = nn_model(x_t, c1, c2, t)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            
            pbar.set_description(f"loss: {loss.item(): .4f}")
            optimizer.step()
            optimizer.zero_grad()
            
        lr_scheduler.step()
            
        # validation, save an image of currently generated samples
        nn_model.eval()
        obj_embedding.eval()
        atr_embedding.eval()
        
        size = x.size()[1:]
        if epoch % args.eval_interval == 0:
            # sample random noise
            x_i = torch.randn(n_samples, *size).to(device)

            # timesteps list
            timesteps = list(range(args.num_timestep))[::-1]

            # create conditions of each class
            c1 = torch.arange(0, args.num_condition[0])
            c2 = torch.arange(0, args.num_condition[1])
            c1 = c1.repeat(args.num_condition[0])
            c2 = c2.repeat(args.num_condition[1])

            c1, c2 = c1.to(device), c2.to(device)

            # conditional embedding, disable label drop
            c1 = obj_embedding(c1, evaluation=True)
            c2 = atr_embedding(c2, evaluation=True)

            # unconditional embedding, create zero embedding vector
            c1_unc = torch.zeros(c1.size(), device=device)
            c2_unc = torch.zeros(c2.size(), device=device)

            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, n_samples))
                temb = time_embedding(t).to(device)
                with torch.no_grad():
                    cond_pred = nn_model(x_i, c1, c2, t)
                    uncond_pred = nn_model(x_i, c1_unc, c2_unc, t)
                    
                    # classifier-free guidance
                    eps = (1 + args.w) * cond_pred - args.w * uncond_pred
                    
                x_i = noise_scheduler.step(eps.cpu(), t[0], x_i.cpu()).to(device)
            
            x0 = x_i.cpu().numpy()
            # unnormalize
            x0 = x0 * 2.5 + 2,5
            plt.plot(x0[:, 0], x0[:], marker=".", linestyle="")
            plt.title(f'Epoch {epoch}')
            plt.savefig(os.path.join('result', f'plot {epoch}.png'))
        
            # save model
            save_root = os.path.join('checkpoints', args.exp)
            os.makedirs(save_root, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'noise_predictor': nn_model.state_dict(),
                'obj_embedding': obj_embedding.state_dict(),
                'atr_embedding': atr_embedding.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_root, f"model_{epoch}.pth"))
        
        
            
                
                    
                
                





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, help='dataset location')
    parser.add_argument('--lr', type=float, defualt=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_worker', type=int, default=2, help='number of workers')
    parser.add_argument('--epochs', type=int, help='total training epochs')
    parser.add_argument('--hidden_size', type=int, help='neurons in fully connected layer')
    parser.add_argument('--hidden_layer', type=int, help='number of hidden layers')
    parser.add_argument('--num_timestep', type=int, help='number of timesteps')
    parser.add_argument('--emb_size', type=int, help='embedding output dimension')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')
    parser.add_argument('--eval_interval', type=int, default=2, help='Frequency of evaluation')
    parser.add_argument('--n_sample', type=int, default=500, help='number of points sampled in evaluation process')
    parser.add_argument('--exp', type=str, default='checkpoints', help='experiment directory name')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    
    args = parser.parse_args()
    
    
    main(args)