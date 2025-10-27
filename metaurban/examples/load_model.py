import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as TF
import torch
import numpy as np

import os
# ===(建议的调试/稳妥环境变量，可保留)===
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0+PTX")
# os.environ.setdefault("NCCL_DEBUG", "INFO")
# os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
# 如果没有 InfiniBand 可开启：
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import importlib
import yaml
import logging
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def trajectory_nms(trajectory_raw, score_raw, max_trajectories=6, distance_threshold=2.0):
    """
    Apply NMS to trajectories based on endpoint distance
    
    Args:
        trajectory_raw: [B, N, T, 3] - batch, num_trajectories, time, (x,y,z)
        score_raw: [B, N] - confidence scores
        max_trajectories: int - number of trajectories to keep
        distance_threshold: float - minimum distance between trajectory endpoints
    
    Returns:
        selected_trajectories: [B, max_trajectories, T, 3]
        selected_scores: [B, max_trajectories]
        selected_indices: [B, max_trajectories]
    """
    B, N, T, _ = trajectory_raw.shape
    
    selected_trajectories = []
    selected_scores = []
    selected_indices = []
    
    for b in range(B):
        # Get trajectories and scores for this batch
        trajs = trajectory_raw[b]  # [N, T, 3]
        scores = score_raw[b]      # [N]
        
        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # NMS selection
        keep_indices = []
        
        for i in sorted_indices:
            if len(keep_indices) >= max_trajectories:
                break
                
            # Check if this trajectory is too close to any kept trajectory
            should_keep = True
            
            for kept_idx in keep_indices:
                # Calculate distance between trajectory endpoints
                endpoint_dist = torch.norm(trajs[i][-1, :2] - trajs[kept_idx][-1, :2])
                
                if endpoint_dist < distance_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i)
        
        # Pad if we don't have enough trajectories
        while len(keep_indices) < max_trajectories:
            keep_indices.append(keep_indices[-1] if keep_indices else 0)
        
        keep_indices = torch.tensor(keep_indices[:max_trajectories])
        
        selected_trajectories.append(trajs[keep_indices])
        selected_scores.append(scores[keep_indices])
        selected_indices.append(keep_indices)
    
    return (torch.stack(selected_trajectories),
            torch.stack(selected_scores))

def load_yaml_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def get_class_from_str(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def min_ade(pred_trajs, gt_trajs, gt_valid_mask, calculate_steps=(9, )) -> float:
    """Compute Average Displacement Error.

    Args:
        pred_trajs: (batch_size, num_modes, pred_len, 2)
        gt_trajs: (batch_size, pred_len, 2)
        gt_valid_mask: (batch_size, pred_len)
    Returns:
        ade: Average Displacement Error

    """
    ade = 0
    for cur_step in calculate_steps:
        dist_error = (pred_trajs[:, :, :cur_step+1, :] - gt_trajs[:, None, :cur_step+1, :]).norm(dim=-1)  # (batch_size, num_modes, pred_len)
        dist_error = (dist_error * gt_valid_mask[:, None, :cur_step+1].float()).sum(dim=-1) / torch.clamp_min(gt_valid_mask[:, :cur_step+1].sum(dim=-1)[:, None], min=1.0)  # (batch_size, num_modes)
        cur_ade = dist_error.min(dim=-1)[0].mean().item()

        ade += cur_ade

    ade = ade / len(calculate_steps)
    return ade

def min_fde(pred_trajs, gt_trajs, gt_valid_mask):
    """Compute minimum Final Displacement Error.
    
    Args:
        pred_trajs: (batch_size, num_modes, pred_len, 2)
        gt_trajs: (batch_size, pred_len, 2)
        gt_valid_mask: (batch_size, pred_len)
        
    Returns:
        fde: Final Displacement Error
    """
    # Get final valid timestep for each sequence
    valid_lengths = gt_valid_mask.sum(dim=-1)  # (batch_size,)
    batch_size = gt_trajs.shape[0]
    
    final_errors = []
    
    for b in range(batch_size):
        final_idx = int(valid_lengths[b]) - 1
        if final_idx >= 0:
            # Final positions
            pred_final = pred_trajs[b, :, final_idx, :]  # (num_modes, 2)
            gt_final = gt_trajs[b, final_idx, :]  # (2,)
            
            # L2 distance for each mode
            dist_error = (pred_final - gt_final).norm(dim=-1)  # (num_modes,)
            final_errors.append(dist_error.min())
    
    return torch.stack(final_errors)


def miss_rate(pred_trajs, gt_trajs, gt_valid_mask, threshold=2.0):
    """Compute Miss Rate - percentage of predictions with FDE > threshold.
    
    Args:
        pred_trajs: (batch_size, num_modes, pred_len, 2)
        gt_trajs: (batch_size, pred_len, 2)
        gt_valid_mask: (batch_size, pred_len)
        threshold: miss threshold in meters (typically 2.0m)
        
    Returns:
        miss_rate: percentage of misses
    """
    valid_lengths = gt_valid_mask.sum(dim=-1)
    batch_size = gt_trajs.shape[0]
    
    misses = []
    
    for b in range(batch_size):
        final_idx = int(valid_lengths[b]) - 1
        if final_idx >= 0:
            pred_final = pred_trajs[b, :, final_idx, :]  # (num_modes, 2)
            gt_final = gt_trajs[b, final_idx, :]  # (2,)
            
            # Best prediction among all modes
            dist_error = (pred_final - gt_final).norm(dim=-1)  # (num_modes,)
            min_error = dist_error.min()
            
            # Check if miss (error > threshold)
            misses.append(min_error > threshold)
    
    if not misses:
        return 0.0
    
    return torch.tensor(misses).float()


def average_precision_at_threshold(pred_trajs, gt_trajs, gt_valid_mask, scores, threshold):
    """Calculate AP at a specific distance threshold."""
    
    all_scores = []
    all_matches = []
    
    batch_size = pred_trajs.shape[0]
    
    for b in range(batch_size):
        # Get final valid timestep
        valid_length = gt_valid_mask[b].sum().item()
        if valid_length == 0:
            continue
            
        final_idx = valid_length - 1
        final_idx = int(final_idx)
        
        # Get predictions and scores for this batch
        pred_final = pred_trajs[b, :, final_idx, :]  # (num_modes, 2)
        gt_final = gt_trajs[b, final_idx, :]         # (2,)
        mode_scores = scores[b, :]                   # (num_modes,)
        
        # Calculate distances
        distances = (pred_final - gt_final).norm(dim=-1)  # (num_modes,)
        
        # Determine matches (distance < threshold)
        matches = (distances < threshold).float()
        
        all_scores.extend(mode_scores.cpu().numpy())
        all_matches.extend(matches.cpu().numpy())
    
    if not all_scores:
        return 0.0
    
    # Convert to numpy arrays
    scores_np = np.array(all_scores)
    matches_np = np.array(all_matches)
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores_np)
    sorted_matches = matches_np[sorted_indices]
    
    # Calculate precision and recall
    tp = np.cumsum(sorted_matches)
    fp = np.cumsum(1 - sorted_matches)
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp + 1e-8)
    
    # Recall = TP / total_positives
    total_positives = np.sum(matches_np)
    if total_positives == 0:
        return 0.0
    
    recall = tp / total_positives
    
    # Calculate AP using trapezoidal rule
    ap = calculate_ap_from_pr(precision, recall)
    
    return ap

def calculate_ap_from_pr(precision, recall):
    """Calculate AP from precision-recall curve."""
    # Add start and end points
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate area under curve
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return ap
class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def load_model():
    # ---- Load + log config ----
    config = load_yaml_config(args.config)

    # ---- Model ----
    model_cls = get_class_from_str(config['model']['model_path'])
    model = model_cls(config['model'])

    # ---- Test ----
    model.eval()

    weights_path = args.ckpt
    checkpoint = torch.load(weights_path, map_location='cpu')
            
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)            

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    model = load_model()
    print(f'Successfully loaded model from {args.ckpt}')
    total_params = sum(p.numel() for p in model.parameters()) / 1024 / 1024
    print(f'Total parameters: {total_params:.2f}M')
    