import torch
import torch.nn as nn

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class SingleObjectTensorEvaluator(object):
    def __init__(self, max_batch_size=25):
        """
        Initializes the evaluator with a maximum batch size to handle large datasets efficiently.
        
        Args:
            max_batch_size (int): The maximum number of samples to process in a single batch. This helps
                                  to manage memory usage and prevent out-of-memory errors.
        """
        self.max_batch_size = max_batch_size
        
    @torch.no_grad()
    def evaluate(self, gt_images, pre_images):
        """
        Evaluates a batch of ground truth and prediction images by calculating various metrics. The inputs
        are expected to be binary masks where pixels of interest are set to 1, and other pixels are set to 0.
        
        Args:
            gt_images (torch.Tensor): The ground truth images. This should be a tensor of shape (N, H, W),
                                      where N is the number of images, and H and W are the height and width
                                      of the images, respectively. The tensor should contain binary values
                                      (0 or 1).
            pre_images (torch.Tensor): The predicted images. This tensor should have the same shape and 
                                       binary format as `gt_images`.
        
        Returns:
            torch.Tensor: A tensor containing the computed metrics (accuracy, precision, recall, IOU, and F1 score).
                          Each metric is represented as a single scalar value.
        """
        
        assert gt_images.shape == pre_images.shape
        assert (gt_images.numel() == (gt_images == 0).sum() + (gt_images == 1).sum())
        assert (pre_images.numel() == (pre_images == 0).sum() + (pre_images == 1).sum())
        
        gt_images = (gt_images > 0).int()
        pre_images = (pre_images > 0).int()
        
        TP = (pre_images & gt_images).sum(dim=(1, 2)).float()
        TN = ((1 - pre_images) & (1 - gt_images)).sum(dim=(1, 2)).float()
        FP = (pre_images & (1 - gt_images)).sum(dim=(1, 2)).float()
        FN = ((1 - pre_images) & gt_images).sum(dim=(1, 2)).float()
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-10) 
        recall = TP / (TP + FN + 1e-10)
        iou = TP / (TP + FP + FN + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10) 
        del TP, TN, FP, FN
        metrics = torch.tensor([
            accuracy.mean().item(),  # Pixel_Accuracy
            precision.mean().item(),
            recall.mean().item(),
            iou.mean().item(),
            f1_score.mean().item()
        ], device=gt_images.device)
        return metrics
    
    def evaluate_in_batches(self, labels, predictions):
        """
        Evaluates large datasets in manageable batches and computes the average metrics across all batches.
        This function manages the iteration over batches and aggregates the results.
        
        Args:
            labels (torch.Tensor): The ground truth labels for the entire dataset. The tensor should be binary
                                   and have a shape of (N, H, W), where N is the total number of images.
            predictions (torch.Tensor): The predicted labels for the entire dataset, with the same format and
                                        shape as `labels`.
        
        Returns:
            torch.Tensor: A tensor containing the average metrics over all batches, aggregated into a single
                          tensor of shape (5,), where each entry corresponds to a different metric.
        """
        device = predictions.device
        num_samples = labels.shape[0]
        nsample = 0
        avg_metrics = torch.zeros(5, device=device)

        for start_idx in range(0, num_samples, self.max_batch_size):
            end_idx = min(start_idx + self.max_batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            batch_labels = labels[start_idx:end_idx]
            batch_predictions = predictions[start_idx:end_idx]
            
            metrics = self.evaluate(batch_labels, batch_predictions)
            
            # Update the running average
            avg_metrics = avg_metrics * (nsample / (nsample + current_batch_size)) + \
                          metrics * (current_batch_size / (nsample + current_batch_size))
            
            nsample += current_batch_size

        return avg_metrics