import os
import time
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from quant import *
from pruner import pruner
from utils import  find_layers
from data_utils import  SA1BEvaluationDataset, SA1BImageDataset, COCOInstanceSegmentation
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, TensorDataset
from feature_utils import DoubleFeatureDataset, FeatureDataset

import logging
from eval import sam_eval

DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def logger_init(log_path, args):
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if args.debug :
        file_handler = logging.FileHandler(os.path.join(log_path, 'debug.log'))
    else:
        file_handler = logging.FileHandler(os.path.join(log_path, '{}_{}.log'.format(args.checkpoint, time.strftime("%Y-%m-%d %H:%M:%S"))))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def intra_block_sparsity_rearrange(layers_dict, initial_sparsity, reduce_layers, increase_layers, theta):
    sparsity_dict = {name: initial_sparsity for name in layers_dict}
    total_pruned_weights_to_move = sum(sparsity_dict[name] * layers_dict[name].weight.numel() * theta for name in reduce_layers)
    for name in reduce_layers:
        pruned_weights = sparsity_dict[name] * layers_dict[name].weight.numel() * theta
        sparsity_dict[name] -= pruned_weights / layers_dict[name].weight.numel()

    total_weights_increase = sum(layers_dict[name].weight.numel() for name in increase_layers)
    for name in increase_layers:
        pruned_weights_increase = (total_pruned_weights_to_move * layers_dict[name].weight.numel()) / total_weights_increase
        sparsity_dict[name] += pruned_weights_increase / layers_dict[name].weight.numel()

    return sparsity_dict

def inter_block_sparsity_rearrange(layers: list[nn.Module], sparsity: float, alpha: float) -> list[float]:
    """
    Adjusts the sparsity across blocks by ensuring the last block does not exceed a specified upper limit, 
    and redistributes any excess sparsity evenly across preceding blocks.
    
    Args:
        layers (list[nn.Module]): List of model layers (blocks).
        sparsity (float): The base sparsity level for all blocks.
        alpha (float): Coefficient for setting the upper limit of the sparsity for the last block. Should be less than 1.
        
    Returns:
        list[float]: Adjusted list of sparsity values for each block.
    """
    sparsity_list = [sparsity] * len(layers)

    if alpha < 1:
        sparsity_upper_limit = sparsity * alpha

        # Check and adjust the sparsity of the last block
        if sparsity_list[-1] > sparsity_upper_limit:
            excess = sparsity_list[-1] - sparsity_upper_limit
            sparsity_list[-1] = sparsity_upper_limit
            increment = excess / (len(sparsity_list) - 1)
            sparsity_list = [sp + increment for sp in sparsity_list[:-1]] + [sparsity_list[-1]]
    
    return sparsity_list

def get_sam(args):
    if args.checkpoint == 'vit_h':
        return sam_model_registry["vit_h"](checkpoint="/path/to/your/sam_vit_h_4b8939.pth")
    elif args.checkpoint == 'vit_l':
        return sam_model_registry["vit_l"](checkpoint="/path/to/your/sam_vit_l_0b3195.pth")
    elif args.checkpoint == 'vit_b':
        return sam_model_registry["vit_b"](checkpoint="/path/to/your/sam_vit_b_01ec64.pth")
    else:
        raise ValueError('Checkpoint not found.')

@torch.no_grad()
def sam_sequential(model, dataloader, dev, logger):
    logger.info('Starting ...')
    layers = model.blocks

    model.patch_embed.to(dev)
    layers[0].to(dev)
    model.pos_embed = nn.Parameter(model.pos_embed.to(dev))
    logger.info('The first layer of sam, Patch Embed, has been moved to device.')

    dtype = next(iter(model.parameters())).dtype
    
    dataloader_iter = iter(dataloader)
    single_batch = next(dataloader_iter)
    feature_shape = model.patch_embed(single_batch.to(dev)).shape
    del single_batch
    
    err_stream_state = torch.zeros(
        (args.nsamples, feature_shape[1], feature_shape[2], feature_shape[3]), dtype=dtype, device=torch.device('cpu')
    )
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.index = 0
        def forward(self, inp):
            err_stream_state[self.index] = inp
            self.index = self.index + 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    sample_count = 0
    for batch in dataloader:
        try:
            model(batch.to(dev)) 
        except ValueError:
            pass
        sample_count += 1
        if sample_count >= args.nsamples:
            break
    layers[0] = layers[0].module
    logger.info('Has callected all samples inputed in the first layer of blocks.')

    
    std_stream_state = err_stream_state.clone()
    std_dataset = FeatureDataset(std_stream_state, device=dev)
    dataloader = DataLoader(std_dataset, batch_size=1, shuffle=False, drop_last=False)
    for idx, std_batch in dataloader:
        std_stream_state[idx] = layers[0](std_batch)[0].cpu()
    
    model.patch_embed.cpu()
    layers[0].cpu()
    model.pos_embed = nn.Parameter(model.pos_embed.cpu())
    
    logger.info('Ready to prune.')
    sparsity_list = inter_block_sparsity_rearrange(layers, args.sparsity, args.alpha)
    window_size = 2
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer.eval()
        sparsity = sparsity_list[i]
        subset = find_layers(layer)
        logger.info('Has found all layers that can be pruned in the {}-th block.'.format(i))
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = pruner(subset[name], str(i) + "." + name)
            sparsity_dict = intra_block_sparsity_rearrange(subset, sparsity, ['attn.qkv'], ['mlp.lin2'], args.beta)

            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
        
        # 1st step
        # This will form the Hessian Metric
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp       
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        err_dataset = FeatureDataset(err_stream_state, device=dev)
        dataloader = DataLoader(err_dataset, batch_size=1, shuffle=False, drop_last=False)
        for idx, err_batch in dataloader:
            layer(err_batch)[0]
        
        for h in handles:
            h.remove()
        logger.info('Has added all samples to calculate the Hessian Metric of all found layers in {}-th block.'.format(i))
        
        # 2nd step
        for name in gpts:
            logger.info('Pruning {} in {}-th block.'.format(name, i))
            gpts[name].fasterprune(
                sparsity_dict[name], prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
        
        # 3rd step
        err_dataset = FeatureDataset(err_stream_state, device=dev)
        dataloader = DataLoader(err_dataset, batch_size=1, shuffle=False, drop_last=False)
        for idx, err_batch in dataloader:
            err_stream_state[idx] = layer(err_batch)[0].cpu()
        
        layer.cpu().eval()
        
        if args.tune_epoch > 0 and i < len(layers) - 1:
            next_layer = layers[i + 1].to(dev)
            
            # 4th step
            std_dataset = FeatureDataset(std_stream_state, device=dev)
            dataloader = DataLoader(std_dataset, batch_size=1, shuffle=False, drop_last=False)
            for idx, std_batch in dataloader:
                std_stream_state[idx] = next_layer(std_batch)[0].cpu()
            
            # 5th step
            logger.info('Make the requires_grad of the blocks to tune True.')
            for param in next_layer.parameters():
                param.requires_grad = True
            next_layer.train()
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(next_layer.parameters(), lr=args.tune_lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.tune_gamma)
            dataset = DoubleFeatureDataset(err_stream_state, std_stream_state, device=dev)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
            scaler = torch.cuda.amp.GradScaler()
            logger.info('Tuning ...')
            for k in range(args.tune_epoch):
                loss_avg = 0
                tmp = 0
                nsample = 0
                for idx, err_batch, std_batch in dataloader: 
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        with torch.cuda.amp.autocast():
                            err_batch = next_layer(err_batch)[0]
                        loss = criterion(err_batch, std_batch)
                        scaler.scale(loss).backward() 
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        tmp = err_batch.shape[0]
                        loss_avg *= nsample / (nsample + tmp)
                        loss_avg += loss.item() * tmp / (nsample + tmp)  
                        nsample += tmp 
                scheduler.step()  
                logger.info('Loss at epoch {}: {}.'.format(k, loss_avg))
            logger.info('End tuning.')
            logger.info('Make the requires_grad of the blocks to tune False.')
            for param in next_layer.parameters():
                param.requires_grad = False
            next_layer.eval()
            gc.collect()
            torch.cuda.empty_cache() 
        gc.collect()
        torch.cuda.empty_cache() 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--batch-size', type=int, default=8, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--sparsity', type=float, default=0, help='Target sparsity')
    parser.add_argument('--prunen', type=int, default=0, help='N for N:M pruning.')
    parser.add_argument('--prunem', type=int, default=0, help='M for N:M pruning.')
    parser.add_argument('--blocksize', type=int, default=128, help='Blocksize to use for adaptive mask selection.')
    parser.add_argument('--wbits', type=int, default=16, help='Whether to quantize as well.')
    parser.add_argument('--minlayer', type=int, default=-1, help='Prune all layers with id >= this.')
    parser.add_argument('--maxlayer', type=int, default=1000, help='Prune all layers with id < this.')
    parser.add_argument('--prune_only', type=str, default='', help='Prune only layers that contain this text.')
    parser.add_argument('--invert', action='store_true', help='Invert subset.')
    parser.add_argument('--save', type=str, default='', help='Path to saved model.')
    parser.add_argument('--interaction_mode', type=str, default='box', choices=['box', 'point', 'mix'], help='Now we have points, box, and mix interaction modes. Choose one based on your need.')
    parser.add_argument('--checkpoint', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='Which checkpoint to use.Chooes in vit_h, vit_l, vit_b.')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode or not.')
    parser.add_argument('--val_dataset', type=str, default='original', choices=['original', 'sa_1b', 'coco'], help='Validation dataset to use. Chooes in original, sa_1b, coco.')
    parser.add_argument("--tune_epoch", type=int, default=0, help="Number of epochs to tune")
    parser.add_argument("--tune_lr", type=float, default=0.00005, help="learing rate to tune")
    parser.add_argument("--tune_gamma", type=float, default=0.85, help="gamma of lr scheduler")
    parser.add_argument("--save_root", type=bool, default=False, help="Save the model or not")
    parser.add_argument("--eval", type=bool, default=True, help="Evaluation or not")
    parser.add_argument("--log_path", type=str, default='./logs', help="Evaluation or not")
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.85, help="sparsity upper limit coefficient")

    args = parser.parse_args()
    
    seed = 120
    set_seed(seed)
    
    logger_path = os.path.join(args.log_path)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
  
    logger = logger_init(logger_path, args)
    
    logger.info('Log has been initialized.')
    logger.info(f'Arguments: {args}')
    
    dataset_root = '/path/to/your/sa_000001' # you need to unzip it first.

    model = get_sam(args)
    logger.info('Sam has been loaded.')
    
    encoder = model.image_encoder
    model.eval()
    encoder.eval()
    logger.info('Sam has been set to evaluation mode.')

    trainset = SA1BImageDataset(root_dir=dataset_root,
                        img_size=encoder.img_size,
                        transform=ResizeLongestSide(encoder.img_size))
    logger.info('Use sa-1b-000001 dataset.')
    
    dataloader = DataLoader(trainset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=1,
                            drop_last=True)
    
    device = torch.device('cuda')
    if (args.sparsity or args.prunen) and not args.debug:
        logger.info('Start to time.')
        tick = time.time()
        sam_sequential(encoder, dataloader, device, logger)
        for n, p in encoder.named_parameters():
            zero_count = torch.sum(p == 0).item()
            param_count = p.numel()
            sparsity = zero_count / param_count
            logger.info('{}: {}'.format(n, sparsity))
            if 'fc2' in n:
                break
        logger.info('End to time. Has used {} seconds.'.format(time.time() - tick))

    if args.save_root:
        save_path = os.path.join(args.save_root, '/{}_tuned/{}.pth'.format(args.model, args.checkpoint))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, os.path.join(logger_path, save_path))
        logger.info('model saved.')
    
    if args.eval:
    
        def my_collate_fn(batch):
            return batch

        logger.info('Start to evaluate.')
        for val_dataset in ['sa_1b', 'coco']:
            if val_dataset == 'original':
                logger.info('Use original dataset.')
                trainset = SA1BEvaluationDataset(root_dir='/path/to/your/sa_000001',  # you need to unzip it first.
                                img_size=encoder.img_size,
                                transform=ResizeLongestSide(encoder.img_size))
            elif val_dataset == 'sa_1b':
                logger.info('Use sa-1b-000003 dataset.')
                trainset = SA1BEvaluationDataset(root_dir='/path/to/your/sa_000003',  # you need to unzip it first.
                                img_size=encoder.img_size,
                                transform=ResizeLongestSide(encoder.img_size),
                                sample=1000)
            elif val_dataset == 'coco':
                logger.info('Use COCO dataset.')
                trainset = COCOInstanceSegmentation(args, base_dir='/path/to/your/COCO',split='val', year='2017')  # path to your COCO dataset root is ok.
            else:
                raise ValueError('Unknown dataset.')
        
            testloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1, collate_fn=my_collate_fn)
                
            torch.cuda.empty_cache()
            sam_eval(model, testloader, device, logger=logger, args=args)
        
