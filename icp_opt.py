import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import gc
import logging
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from quant import *
from pruner import pruner
from utils import find_layers
from feature_utils import DoubleFeatureDataset, FeatureDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_opt(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM  # type: ignore
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.float32)  # torch_dtype='auto'
    model.seqlen = model.config.max_position_embeddings
    return model

def logger_init(log_path, args):
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if args.debug :
        file_handler = logging.FileHandler(os.path.join(log_path, 'debug.log'))
    else:
        file_handler = logging.FileHandler(os.path.join(log_path, '{}_{}.log'.format(args.model.split("/")[-1], time.strftime("%Y-%m-%d %H:%M:%S"))))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def intra_block_sparsity_rearrange(layers_dict: dict[str, nn.Module], initial_sparsity: float, reduce_layers: list[str], increase_layers: list[str], beta: float) -> dict[str, float]:
    """
    Adjusts the sparsity of different layers by redistributing pruning quotas within a block.
    
    Args:
        layers_dict (dict[str, nn.Module]): Dictionary of layers to adjust.
        initial_sparsity (float): Initial sparsity value for all layers.
        reduce_layers (list[str]): Layers from which to reduce pruning.
        increase_layers (list[str]): Layers to which to increase pruning.
        theta (float): Redistribution factor for pruning weights.
        
    Returns:
        dict[str, float]: Dictionary of adjusted sparsity values for each layer.
    """
    # Initialize sparsity for each layer with the initial value
    sparsity_dict = {name: initial_sparsity for name in layers_dict}

    # Calculate the total number of pruned weights to move from reduce_layers
    total_pruning_quota_to_move = sum(sparsity_dict[name] * layers_dict[name].weight.numel() * beta for name in reduce_layers)

    # Update sparsity in reduce_layers
    for name in reduce_layers:
        pruned_weights = sparsity_dict[name] * layers_dict[name].weight.numel() * beta
        sparsity_dict[name] -= pruned_weights / layers_dict[name].weight.numel()

    # Increase sparsity in increase_layers proportionally to their weight counts
    total_increase_layer_weights = sum(layers_dict[name].weight.numel() for name in increase_layers)
    
    # Update sparsity in increase_layers
    for name in increase_layers:
        pruning_quota_addition = (total_pruning_quota_to_move * layers_dict[name].weight.numel()) / total_increase_layer_weights
        sparsity_dict[name] += pruning_quota_addition / layers_dict[name].weight.numel()

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

@torch.no_grad()
def opt_sequential(model, dataloader, dev, logger):
    logger.info('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    err_stream_state = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=torch.device('cpu')
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            err_stream_state[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev)) 
        except ValueError:
            pass
    layers[0] = layers[0].module

    attention_mask = cache['attention_mask']
    
    std_stream_state = err_stream_state.clone()  # 2048M opt-1.3b
    std_dataset = FeatureDataset(std_stream_state, device=dev)
    dataloader = DataLoader(std_dataset, batch_size=1, shuffle=False, drop_last=False)
    for idx, std_batch in dataloader:
        std_stream_state[idx] = layers[0](std_batch, attention_mask=attention_mask)[0].cpu()

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    logger.info('Ready.')
    
    sparsity_list = inter_block_sparsity_rearrange(layers, args.sparsity, args.alpha)
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)  # 一个block 200M opt-1.3b
        sparsity = sparsity_list[i]
        subset = find_layers(layer)
        
        # "qkvof1-f2", "qkvo-f2", "qkvo-f1", "qkvo-f1f2", "qkv-f2", "qkv-f1", "qkv-o", "qkv-of1", "qkv-of2", "qkv-of1f2"
        if args.rearrange_mode == "qkvof1-f2":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', "fc1"]  # 减少剪枝的层
            increase_layers = ['fc2']
        elif args.rearrange_mode == "qkvo-f2":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']
            increase_layers = ['fc2']
        elif args.rearrange_mode == "qkv-f2":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['fc2']
        elif args.rearrange_mode == "qkvo-f1f2":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']
            increase_layers = ['fc1', 'fc2']
        elif args.rearrange_mode == "qkv-of1f2":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.out_proj', 'fc1', 'fc2']
        elif args.rearrange_mode == "qkvo-f1":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']
            increase_layers = ['fc1']
        elif args.rearrange_mode == "qkv-f1":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['fc1']
        elif args.rearrange_mode == "qkv-o":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.out_proj']
        elif args.rearrange_mode == "qkv-of1":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.out_proj', 'fc1']
        elif args.rearrange_mode == "qkv-of2":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.out_proj', 'fc2']
        else:
            raise ValueError("Invalid rearrange mode.")
        
        beta = args.beta
        sparsity_dict = intra_block_sparsity_rearrange(subset, sparsity, reduce_layers, increase_layers, beta)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = pruner(subset[name], str(i) + "." + name)
            # gpts[name] = pruner(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
        
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
            layer(err_batch, attention_mask=attention_mask)[0]
        
        for h, name in zip(handles, gpts):
            h.remove()
            # gpts[name].hist_out()
        logger.info('Has added all samples to calculate the Hessian Metric of all found layers in {}-th layer.'.format(i))
        
        for name in gpts:
            logger.info('Pruning {} in {}-th layer.'.format(name, i))
            gpts[name].fasterprune(
                sparsity_dict[name], prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        err_dataset = FeatureDataset(err_stream_state, device=dev)
        dataloader = DataLoader(err_dataset, batch_size=1, shuffle=False, drop_last=False)
        for idx, err_batch in dataloader:
            err_stream_state[idx] = layer(err_batch, attention_mask=attention_mask)[0].cpu()

        layers[i] = layer.cpu()
        
        if args.tune_epoch > 0 and i < len(layers) - 1:
            rest_layer = layers[i + 1].to(dev)
            
            std_dataset = FeatureDataset(std_stream_state, device=dev)
            dataloader = DataLoader(std_dataset, batch_size=1, shuffle=False, drop_last=False)
            for idx, std_batch in dataloader:
                std_stream_state[idx] = rest_layer(std_batch, attention_mask=attention_mask)[0].cpu()
                
            logger.info('Make the requires_grad of the layers to tune True.')
            for param in rest_layer.parameters():
                param.requires_grad = True
            rest_layer.train()
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(rest_layer.parameters(), lr=args.tune_lr)
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
                            err_batch = rest_layer(err_batch, attention_mask=attention_mask)[0]
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
            logger.info('Make the requires_grad of the layers to tune False.')
            for param in rest_layer.parameters():
                param.requires_grad = False
            rest_layer.eval()
            gc.collect()
            torch.cuda.empty_cache() 
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, logger):
    logger.info('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    logger.info(f"*****************************************************************")
    logger.info(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from data_utils import get_loaders

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb','c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--sparsity', type=float, default=0, help='Target sparsity.')
    parser.add_argument('--prunen', type=int, default=0, help='N for N:M pruning.')
    parser.add_argument('--prunem', type=int, default=0, help='M for N:M pruning.')
    parser.add_argument('--blocksize', type=int, default=128, help='Blocksize to use for adaptive mask selection.')
    parser.add_argument('--gmp', action='store_true', help='Whether to run the GMP baseline.')
    parser.add_argument('--wbits', type=int, default=16, help='Whether to quantize as well.')
    parser.add_argument('--minlayer', type=int, default=-1, help='Prune all layers with id >= this.')
    parser.add_argument('--maxlayer', type=int, default=1000, help='Prune all layers with id < this.')
    parser.add_argument('--prune_only', type=str, default='', help='Prune only layers that contain this text.')
    parser.add_argument('--invert', action='store_true', help='Invert subset.')
    parser.add_argument('--save', type=str, default=None, help='Path to saved model.')
    parser.add_argument('--log_path', type=str, default='./logs_lang', help='Path to log fold.')
    parser.add_argument('--tune_epoch', type=int, default=0, help='Number of epochs to tune.')
    parser.add_argument('--tune_lr', type=float, default=0.00005, help='Learning rate to tune.')
    parser.add_argument('--tune_gamma', type=float, default=0.87, help='Gamma of lr scheduler.')
    parser.add_argument('--tune_beta', type=float, default=1, help='Learning rate gradually increases with each training session.')
    parser.add_argument('--alpha', type=float, default=0.85, help='Sparsity upper limit coefficient.')
    parser.add_argument('--beta', type=float, default=0, help='Inter-block sparsity rearrangement parameter.')
    parser.add_argument('--rearrange_mode', type=str, default="qkv-f2", choices=["qkvof1-f2", "qkvo-f2", "qkvo-f1", "qkvo-f1f2", "qkv-f2", "qkv-f1", "qkv-o", "qkv-of1", "qkv-of2", "qkv-of1f2"], help='rearrange mode.')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode or not.')

    args = parser.parse_args()

    seed = 120
    set_seed(seed)
    
    logger_path = os.path.join(args.log_path)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = logger_init(logger_path, args)
    logger.info('Log has been initialized.')
    logger.info(f'Arguments: {args}')
    
    model = get_opt(args.model)
    logger.info('{} has been loaded.'.format(args.model))
    
    model.eval()
    logger.info('{} has been set to evaluation mode.'.format(args.model))

    dataloader, testloader, tokenizer = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    logger.info('Dataloader and testloader of {} have been initialized.'.format(args.dataset))
    DEV = torch.device('cuda')
    if (args.sparsity or args.prunen) and not args.gmp:
        logger.info('Start to time.')
        tick = time.time()
        opt_sequential(model, dataloader, DEV, logger)
        total_zero = 0
        total_params = 0
        for n, p in model.model.decoder.layers.named_parameters():
            zero_count = torch.sum(p == 0).item()
            param_count = p.numel()
            sparsity = zero_count / param_count
            logger.info('{}: {}'.format(n, sparsity))
            total_zero += zero_count
            total_params += param_count

        overall_sparsity = total_zero / total_params
        logger.info('Overall sparsity: {}'.format(overall_sparsity))
            # if 'fc2' in n:
            #     break
        logger.info('End to time. Has used {} seconds.'.format(time.time() - tick))

    for dataset in ['wikitext2', 'ptb', 'c4']:
    # for dataset in ['wikitext2', 'ptb']:
        dataloader, testloader, tokenizer = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        logger.info("Start to evaluate on {}.".format(dataset))
        opt_eval(model, testloader, DEV, dataset, logger)

    if args.save:
        model.save_pretrained(args.save)
