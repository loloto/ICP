import fnmatch

def llama_eval_zero_shot(model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, device="cuda:0"):
    from lm_eval import tasks, evaluator
    from lm_eval.models.huggingface import HFLM
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_manager = tasks.TaskManager()
    task_names = pattern_match(task_list, task_manager.all_tasks)
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer)
    limit = None 
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=task_names,
        device=device,
        cache_requests=True,
        limit=limit
    )

    return results 

import torch
import numpy as np
from tqdm import tqdm
from segment_anything import SamPredictor
from utils import SingleObjectTensorEvaluator, find_layers



@torch.no_grad()
def sam_eval(model, 
             dataloader, 
             dev, 
             args,
             logger=None, ):
    model.to(dev)
    predictor = SamPredictor(model)
    evaluator = SingleObjectTensorEvaluator()
    
    if args.pruning_mode == 'magnitude':
        logger.info('Pruning the model based on magnitude.')
        blocks = model.image_encoder.blocks
        for i in range(len(blocks)):
            # block = blocks[i].to(dev)
            block = blocks[i]

            subset = find_layers(block)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0
            
            del block
            torch.cuda.empty_cache()
    
    nsample = 0
    avg_metrics = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    cache_limited_constant = 1500 * 2000 * 100  # This is to fully utilize the video memory, and the constant is independent of the inference result (3090 + H model + SA-1B)
    for i, sample in pbar:
        sample = sample[0]
        img_array = np.array(sample['image'])
        predictor.set_image(img_array, 'RGB')
        anno_batch_size = int(cache_limited_constant / (img_array.shape[0] * img_array.shape[1]))

        if len({len(sample[k]) for k in ['label', 'points', 'bboxes']}) != 1:
            raise ValueError("Data loading error: Lengths of cls, label, points, and bboxes are not equal.")
        
        for start_idx in range(0, len(sample['bboxes']), anno_batch_size):
            
            end_idx = min(start_idx + anno_batch_size, len(sample['bboxes']))
            
            if args.interaction_mode == 'point':
                points = np.array(sample['points'][start_idx:end_idx])
                masks, iou_pred, _ = predictor.predict(point_coords=points, point_labels=np.ones(points.shape[:-1]), multimask_output=True, multi_sample_input=False)

            elif args.interaction_mode == 'box':
                bboxes = np.array(sample['bboxes'][start_idx:end_idx])
                bboxes[:, 2:] += bboxes[:, :2]
                masks, iou_pred, _ = predictor.predict(box=bboxes, multimask_output=True, multi_sample_input=False)

            elif args.interaction_mode == 'mix':
                continue
            else:
                raise ValueError("Invalid interaction mode. You have to choose from 'point', 'box', 'mix'.")
            
            
            mask = masks[torch.arange(masks.shape[0]), torch.argmax(iou_pred, dim=1)]
            metrics = evaluator.evaluate_in_batches(sample['label'][start_idx:end_idx].long().to(mask.device), mask.long())
            tmp = end_idx - start_idx
            avg_metrics *= nsample / (tmp + nsample)
            avg_metrics += tmp / (tmp + nsample) * metrics
            nsample += tmp
        
        pbar.set_description(f"PA: {avg_metrics[0].item():.4f}, IoU: {avg_metrics[3].item():.4f}, Precision: {avg_metrics[1].item():.4f}, Recall: {avg_metrics[2].item():.4f}, F1: {avg_metrics[4].item():.4f}")
        
        if args.debug:
            break
    logger.info(f"*****************************************************************")
    logger.info(f"Pixel Accuracy of the dataset: {avg_metrics[0].item()}")
    logger.info(f"Intersection over Union: {avg_metrics[3].item()}")
    logger.info(f"Precision: {avg_metrics[1].item()}")
    logger.info(f"Recall: {avg_metrics[2].item()}")
    logger.info(f"F1 Score: {avg_metrics[4].item()}")
    logger.info('End of evaluation.')