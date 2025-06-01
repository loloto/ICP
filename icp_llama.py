import os
import time

import torch
import torch.amp
import torch.nn as nn
import random
import numpy as np

from pruner import *
# from modelutils import *
from quant import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from eval import llama_eval_zero_shot
from data_utils import get_loaders
from utils import find_layers
import logging
from feature_utils import DoubleFeatureDataset, FeatureDataset
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
    model.seqlen = 2048
    return model

def adjust_sparsity_weighted(layers_dict, initial_sparsity, reduce_layers, increase_layers, theta):
    # 初始化每层的稀疏度为统一的初始值
    sparsity_dict = {name: initial_sparsity for name in layers_dict}

    # 计算要从reduce_layers中移除的总剪枝权重数量
    total_pruned_weights_to_move = sum(sparsity_dict[name] * layers_dict[name].weight.numel() * theta for name in reduce_layers)

    # 更新reduce_layers中的稀疏度
    for name in reduce_layers:
        pruned_weights = sparsity_dict[name] * layers_dict[name].weight.numel() * theta
        sparsity_dict[name] -= pruned_weights / layers_dict[name].weight.numel()

    # 按权重数量比例增加increase_layers中的稀疏度
    total_weights_increase = sum(layers_dict[name].weight.numel() for name in increase_layers)
    for name in increase_layers:
        pruned_weights_increase = (total_pruned_weights_to_move * layers_dict[name].weight.numel()) / total_weights_increase
        sparsity_dict[name] += pruned_weights_increase / layers_dict[name].weight.numel()

    return sparsity_dict


@torch.no_grad()
def llama_sequential(model, dataloader, dev, logger):
    logger.info('Starting...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    err_stream_state = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=torch.device('cpu')
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            err_stream_state[cache["i"]] = inp.cpu()
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    
    std_stream_state = err_stream_state.clone()
    std_dataset = FeatureDataset(std_stream_state, device=dev)
    dataloader = DataLoader(std_dataset, batch_size=1, shuffle=False, drop_last=False)
    for idx, std_batch in dataloader:
        std_stream_state[idx] = layers[0](std_batch, attention_mask=attention_mask, position_ids=position_ids)[0].cpu()
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    logger.info('Ready.')
    
    sparsity_list = [args.sparsity] * len(layers)
    
    if args.alpha < 1:
        # 设置最后一个block的稀疏度上限系数参数
        sparsity_upper_limit_coefficient = args.alpha  # 系数应小于1
        sparsity_upper_limit = args.sparsity * sparsity_upper_limit_coefficient

        # 检查并调整最后一个block的稀疏度
        if sparsity_list[-1] > sparsity_upper_limit:
            excess = sparsity_list[-1] - sparsity_upper_limit
            sparsity_list[-1] = sparsity_upper_limit
            
            # 将超出部分平均分配到前面的元素，除了第一个block
            increment = excess / (len(sparsity_list) - 1)
            sparsity_list = [sp + increment for sp in sparsity_list[:-1]] + [sparsity_list[-1]]

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

                # "qkvogu-d", "qkvo-d", "qkvo-gu", "qkvo-gud", "qkv-d", "qkv-gu", "qkv-o", "qkv-ogu", "qkv-od", "qkv-ogud"
        if args.rearrange_mode == "qkvogu-d":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj']
            increase_layers = ['mlp.down_proj']
        elif args.rearrange_mode == "qkvo-d":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
            increase_layers = ['mlp.down_proj']
        elif args.rearrange_mode == "qkv-d":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['mlp.down_proj']
        elif args.rearrange_mode == "qkvo-gud":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
            increase_layers = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        elif args.rearrange_mode == "qkv-ogud":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        elif args.rearrange_mode == "qkvo-gu":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
            increase_layers = ['mlp.gate_proj', 'mlp.up_proj']
        elif args.rearrange_mode == "qkv-gu":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['mlp.gate_proj', 'mlp.up_proj']
        elif args.rearrange_mode == "qkv-o":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.o_proj']
        elif args.rearrange_mode == "qkv-ogu":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj']
        elif args.rearrange_mode == "qkv-od":
            reduce_layers = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
            increase_layers = ['self_attn.o_proj', 'mlp.down_proj']
        else:
            raise ValueError("Invalid rearrange mode.")

        beta = args.beta
        sparsity_dict = adjust_sparsity_weighted(full, args.sparsity, reduce_layers, increase_layers, beta)
        ## 写到这没写完
        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = pruner(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))  
            err_dataset = FeatureDataset(err_stream_state, device=dev)
            dataloader = DataLoader(err_dataset, batch_size=1, shuffle=False, drop_last=False)
            for idx, err_batch in dataloader:
                layer(err_batch, attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            logger.info('Has added all samples to calculate the Hessian Metric of all found layers in {}-th layer.'.format(i))
            
            for name in subset:
                logger.info('Pruning {} in {}-th layer, target sparse ratio: '.format(name, i, sparsity_dict[name]))
                gpts[name].fasterprune(
                    sparsity_dict[name],
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

        err_dataset = FeatureDataset(err_stream_state, device=dev)
        dataloader = DataLoader(err_dataset, batch_size=1, shuffle=False, drop_last=False)
        for idx, err_batch in dataloader:
            err_stream_state[idx] = layer(err_batch, attention_mask=attention_mask, position_ids=position_ids)[0].cpu()
        
        layers[i] = layer.cpu()
        
        if args.tune_epoch > 0 and i < len(layers) - 1:
            rest_layer = layers[i + 1].to(dev)

            std_dataset = FeatureDataset(std_stream_state, device=dev)
            dataloader = DataLoader(std_dataset, batch_size=1, shuffle=False, drop_last=False)
            for idx, std_batch in dataloader:
                std_stream_state[idx] = rest_layer(std_batch, attention_mask=attention_mask, position_ids=position_ids)[0].cpu()
        
            for param in rest_layer.parameters():
                param.requires_grad = True
            rest_layer.train()
        
            criterion = nn.MSELoss()
            optimizer = optim.Adam(rest_layer.parameters(), lr=args.tune_lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.tune_gamma)
            dataset = DoubleFeatureDataset(err_stream_state, std_stream_state, device=dev)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)
            scaler = torch.amp.GradScaler()
            logger.info('Tuning ...')
            for k in range(args.tune_epoch):
                loss_avg = 0
                tmp = 0
                nsample = 0
                for idx, err_batch, std_batch in dataloader: 
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        with torch.cuda.amp.autocast():
                            err_batch = rest_layer(err_batch, attention_mask=attention_mask, position_ids=position_ids)[0]
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
            for param in rest_layer.parameters():
                param.requires_grad = False
            rest_layer.eval()
            
        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev, logger, dataset: str, log_wandb: bool = False):
    logger.info("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=torch.device('cpu'))
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        logger.info(f"Pruning layer {i}")
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0
        inps_dataset = FeatureDataset(inps, device=dev)
        dataloader = DataLoader(inps_dataset, batch_size=1, shuffle=False, drop_last=False)
        for idx, batch in dataloader:
            inps[idx] = layer(batch, attention_mask=attention_mask, position_ids=position_ids)[0].cpu()
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    inps_dataset = FeatureDataset(inps, device=dev)
    dataloader = DataLoader(inps_dataset, batch_size=1, shuffle=False, drop_last=False)
    for idx, inp in dataloader:
        # hidden_states = inp.unsqueeze(0)
        hidden_states = inp
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (idx * model.seqlen) : ((idx + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    logger.info(f"*****************************************************************")
    logger.info(f"Perplexity: {ppl.item():3f}")
    logger.info(f"*****************************************************************")
    
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    # from datautils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"], help="Where to extract calibration data from.")
    parser.add_argument("--seed", type=int, default=120, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening.")
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument("--blocksize", type=int, default=128, help="Blocksize to use for adaptive mask selection.")
    parser.add_argument("--gmp", action="store_true", help="Whether to run the GMP baseline.")
    parser.add_argument("--wbits", type=int, default=16, help="Whether to quantize as well.")
    parser.add_argument("--minlayer", type=int, default=-1, help="Prune all layers with id >= this.")
    parser.add_argument("--maxlayer", type=int, default=1000, help="Prune all layers with id < this.")
    parser.add_argument("--prune_only", type=str, default="", help="Prune only layers that contain this text.")
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument("--true-sequential", action="store_true", help="Whether to run in true sequential model.")
    parser.add_argument("--log_wandb", action="store_true", help="Whether to log to wandb.")
    parser.add_argument('--log_path', type=str, default='./logs_lang_llama', help='Path to log fold.')
    parser.add_argument('--tune_epoch', type=int, default=0, help='Number of epochs to tune.')
    parser.add_argument('--tune_lr', type=float, default=0.00005, help='Learning rate to tune.')
    parser.add_argument('--tune_gamma', type=float, default=0.87, help='Gamma of lr scheduler.')
    parser.add_argument('--tune_beta', type=float, default=1, help='Learning rate gradually increases with each training session.')
    parser.add_argument('--alpha', type=float, default=1, help='Sparsity upper limit coefficient.')
    parser.add_argument('--beta', type=float, default=0, help='Inter-block sparsity rearrangement parameter.')
    parser.add_argument('--rearrange_mode', type=str, default="qkv-d", choices=["qkvogu-d", "qkvo-d", "qkvo-gu", "qkvo-gud", "qkv-d", "qkv-gu", "qkv-o", "qkv-ogu", "qkv-od", "qkv-ogud"], help='rearrange mode.')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode or not.')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)
    logger_path = os.path.join(args.log_path)
    
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = logger_init(logger_path, args)
    logger.info('Log has been initialized.')
    logger.info(f'Arguments: {args}')
    
    model = get_llama(args.model)
    model.eval()
    # model.to(DEV)  # 需删掉，此处仅为测试
    dataloader, testloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        logger.info('Start to time.')
        tick = time.time()
        llama_sequential(model, dataloader, DEV, logger)
        for n, p in model.named_parameters():
            logger.info('{}: {}'.format(n, torch.mean((p == 0).float())))
            if 'down_proj' in n:
                break
        logger.info('End to time.')

    total_zero = 0
    total_params = 0
    for n, p in model.model.layers.named_parameters():
        zero_count = torch.sum(p == 0).item()
        param_count = p.numel()
        sparsity = zero_count / param_count
        logger.info('{}: {}'.format(n, sparsity))
        total_zero += zero_count
        total_params += param_count
    overall_sparsity = total_zero / total_params
    logger.info('Overall sparsity: {}'.format(overall_sparsity))
    
    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader, tokenizer = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        model.eval()
        logger.info("Dataset: {}".format(dataset))
        llama_eval(model, testloader, DEV, logger=logger, dataset=dataset, log_wandb=args.log_wandb)
    model.half()
    model.to(DEV)
    task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    num_shot = 0
    results = llama_eval_zero_shot(model, tokenizer, task_list, num_shot, device=DEV)
    
    # 打印出task_list中每个任务的准确率
    logger.info("********************************")
    logger.info('Zero-shot Evaluation Results (Accuracy)')

    accuracies = []
    for task in task_list:
        if task in results['results'].keys():
            acc = results['results'][task].get('acc,none', 'N/A')
            logger.info(f"{task}: Accuracy = {acc}")
            accuracies.append(acc)
        else:
            logger.info(f"{task}: No results available")
    avg_accuracy = sum(accuracies) / len(accuracies)
    logger.info(f"Average Accuracy = {avg_accuracy:.4f}")
    logger.info("********************************")
    
    task_list = ["mmlu"]
    num_shot = 5
    results = llama_eval_zero_shot( model, tokenizer, task_list, num_shot, device=DEV)
    logger.info("********************************")
    logger.info("5-shot Evaluation Results (Accuracy)")
    accuracies = []
    for task in task_list:
        if task in results['results'].keys():
            acc = results['results'][task].get('acc,none', 'N/A')
            logger.info(f"{task}: Accuracy = {acc}")
        else:
            logger.info(f"{task}: No results available")
    logger.info("********************************")

    if args.save:
        model.save_pretrained(args.save)
