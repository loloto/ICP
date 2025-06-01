import os
import random
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, LlamaTokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def load_or_download_dataset(dataset_root, dataset_name, config_name, split_name):
    if os.path.exists(os.path.join(dataset_root, split_name)):
        return load_from_disk(os.path.join(dataset_root, split_name))
    else:
        os.makedirs(os.path.join(dataset_root, split_name))
        dataset = load_dataset(dataset_name, config_name, split=split_name)
        dataset.save_to_disk(os.path.join(dataset_root, split_name))
        return dataset

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', download_mode='reuse_cache_if_exists')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', download_mode='reuse_cache_if_exists')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        dataloader, testloader = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        dataloader, testloader = get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        dataloader, testloader = get_c4(nsamples, seed, seqlen, tokenizer)
    return dataloader, testloader, tokenizer

#####################################
# Vision Datasets
#####################################
import json
from PIL import Image
from itertools import repeat
from torch.utils.data import Dataset
from torch.nn import functional as F
from pycocotools.mask import decode
from segment_anything.utils.transforms import ResizeLongestSide
from typing import List


class SA1BEvaluationDataset(Dataset):
    def __init__(self, root_dir, img_size=1024, transform=ResizeLongestSide, sample=None):
        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            img_size (int, optional): Desired image size after resizing. Default is 1024.
            transform (callable, optional): Optional transform to be applied on both image and mask.
            sample (int, optional): If provided, randomly samples the specified number of images from the dataset for evaluation.
        """
        self.root_dir = root_dir
        self.pth_dir = os.path.join(self.root_dir, 'pth')
        self.transform = transform
        self.transform = transform
        self.img_size = img_size
        self.image_names = sorted([f.split('.')[0] for f in os.listdir(root_dir) if f.endswith('.jpg')])
        
        if sample:
            self.image_names = random.sample(self.image_names, min(sample, len(self.image_names)))
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(os.path.join(self.root_dir, image_name + '.jpg')).convert('RGB')
        
        json_file = os.path.join(self.root_dir, image_name + '.json')
        
        point_coords = list()
        point_labels = list()
        bboxes = list()
        label = list()
        with open(json_file, 'r') as file:
            data = json.load(file)
        for ann in data['annotations']:
            point_coords.append(ann['point_coords'])
            point_labels.append([1])
            bboxes.append(ann['bbox'])
            label.append(ann['segmentation'])
            
        sample = {
            'cls': repeat(None, len(label)),
            'image' : image,
            'image_id': data['image']['image_id'],
            'label': torch.tensor(decode(label)).permute(2, 0, 1),
            'points' : torch.tensor(point_coords),
            'bboxes' : torch.tensor(bboxes),
            }
    
        return sample
    
class SA1BImageDataset(Dataset):
    def __init__(self, root_dir, img_size=1024, transform=ResizeLongestSide, 
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375],):
        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            transform (callable, optional): Optional transform to be applied on both image and mask.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transform
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.img_size = img_size

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        input_image = np.array(Image.open(img_name).convert('RGB'))
        
        # Apply the same transform to both image and mask
        input_image = self.transform.apply_image(input_image)
        input_image_torch = torch.as_tensor(input_image, device='cpu')
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        input_image_torch = self.preprocess(input_image_torch)
        
        return input_image_torch
    
from pycocotools.coco import COCO
from pycocotools import mask
from tqdm import trange

class COCOInstanceSegmentation(Dataset):
    
    def __init__(self,
                 args,
                 base_dir,
                 point_num=5,
                 split='train',
                 year='2017',
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375],):
        super().__init__()
        
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        valid_ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, '{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.transform = ResizeLongestSide(1024)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.img_size = 1024
        self.point_num = point_num
        self.CAT_LIST = list(self.coco.cats.keys())

        if os.path.exists(valid_ids_file):
            self.valid_ids = torch.load(valid_ids_file)
        else:
            self.valid_ids = self._preprocess(valid_ids_file)
        self.args = args

    def __getitem__(self, index):
        _img, cls, _target, img_id, bboxes, points = self._make_img_gt_bbox_pair(index)
        sample = {'img_id': img_id, 'image': _img, 'label': _target, 'cls': cls, 'points': points, 'bboxes': bboxes}

        return sample

    def _make_img_gt_bbox_pair(self, index):
        img_id = list(self.valid_ids.keys())[index]
        img_metadata = self.coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        valid_ann_ids = self.valid_ids[img_id]
        cocotarget = self.coco.loadAnns(valid_ann_ids)
        
        cls_tensor = torch.empty(0, dtype=torch.int64)
        masks_tensor = torch.empty((0, img_metadata['height'], img_metadata['width']), dtype=torch.uint8)
        bboxes_tensor = torch.empty((0, 4))
        points_tensor = torch.empty((0, self.point_num, 2))
        coco_mask = self.coco_mask
        
        for instance in cocotarget:
            if instance['category_id'] not in self.CAT_LIST:
                continue
            rle = coco_mask.frPyObjects(instance['segmentation'], img_metadata['height'], img_metadata['width'])
            m = coco_mask.decode(rle)
            if len(m.shape) == 3:
                m = m.max(axis=2)  # 如果存在多个层面，合并它们
            mask = m.astype(np.uint8)
            
            points = self.select_random_points(mask, self.point_num)
            
            masks_tensor = torch.cat((masks_tensor, torch.tensor(mask).unsqueeze(0)), dim=0)
            bboxes_tensor = torch.cat((bboxes_tensor, torch.tensor(instance['bbox']).unsqueeze(0)), dim=0)
            points_tensor = torch.cat((points_tensor, torch.tensor(points).unsqueeze(0)), dim=0)
            cls_tensor = torch.cat((cls_tensor, torch.tensor(instance['category_id']).unsqueeze(0)), dim=0)

        return _img, cls_tensor, masks_tensor, img_id, bboxes_tensor, points_tensor

    def _preprocess(self, valid_anns_file):
        print("Preprocessing annotations, this might take a while, but it only runs once for each split.")
        valid_ann_ids = {}
        tbar = trange(len(self.coco.getImgIds()))
        for i in tbar:
            img_id = self.coco.getImgIds()[i]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            valid_anns = [ann['id'] for ann in anns if self.is_valid_ann(ann, img_id)]
            if valid_anns:
                valid_ann_ids[img_id] = valid_anns
            tbar.set_description(f'Processing: {i + 1}/{len(self.coco.getImgIds())}, valid images found: {len(valid_ann_ids)}')
        torch.save(valid_ann_ids, valid_anns_file)
        return valid_ann_ids
    
    def is_valid_ann(self, ann, img_id):
        '''Check if the annotation has at least a valid 0-1 mask and a bbox'''
        if 'segmentation' in ann:
            img_metadata = self.coco.loadImgs(img_id)[0]
            rle = self.coco_mask.frPyObjects(ann['segmentation'], img_metadata['height'], img_metadata['width'])
            mask = self.coco_mask.decode(rle)
            if mask.any() and np.isin(mask, [0, 1]).all():
                return True
        return False

    def select_random_points(self, mask, n):
        """
        在一个0-1掩码中随机选择n个点。
        
        参数:
        - mask: np.array，是一个0-1的掩码，其中1表示特定实例的区域。
        - n: int，要从掩码中随机选择的点的数量。
        
        返回:
        - points: 一个N×2 numpy数组，包含随机选中的点的坐标。
        
        异常:
        - ValueError: 如果掩码不是一个有效的0-1掩码或者没有任何1值，即没有有效点可选择。
        """
        
       
        # 检查掩码是否严格只包含0和1，并且至少包含一个1
        if not np.isin(mask, [0, 1]).all() or np.all(mask == 0):
            raise ValueError("Mask must be a binary 0-1 mask with at least one '1' value.")
        
        # 找到所有值为1的坐标点
        y_coords, x_coords = np.where(mask == 1)
        
        # 如果可用点的数量小于n，则允许重复选择；如果足够，则不允许重复
        allow_repeats = len(x_coords) < n
        random_indices = np.random.choice(len(x_coords), n, replace=allow_repeats)
        points = np.column_stack((x_coords[random_indices], y_coords[random_indices]))
        
        return points

    def __len__(self):
        return len(self.valid_ids)


  