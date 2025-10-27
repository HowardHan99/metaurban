"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""

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
    
from metaurban import SidewalkCleanMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
import cv2
import numpy as np
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.engine.logger import get_logger
import argparse
import torch
"""
Block Type	    ID
Straight	    S  
Circular	    C   #
InRamp	        r   #
OutRamp	        R   #
Roundabout	    O	#
Intersection	X
Merge	        y	
Split	        Y   
Tollgate	    $	
Parking lot	    P.x
TInterection	T	
Fork	        WIP
"""

import cv2
import numpy as np

def project_waypoints_to_fisheye_image_with_polygon(
        waypoints, intrinsic_params, image_path, save_path='projected_waypoints.jpg', 
        camera_height=0.41, save_fig=False, color=None):

    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints should be of shape (N, 2)")
    if len(intrinsic_params) != 4:
        raise ValueError("intrinsic_params should be [f, cx, cy, k]")
    
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    else:
        print(f'Unsupported image_path type: {type(image_path)}')

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    H, W = img.shape[:2]
    f, cx, cy, k = intrinsic_params

    # ========== step1: 生成左右偏移的waypoints ==========
    # Make the path wider to match navigation style
    path_width = 0.22  # Increased width for navigation path
    offsets_left, offsets_right = [], []
    
    for i in range(len(waypoints)-1):
        p1, p2 = waypoints[i], waypoints[i+1]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        dir_unit = direction / norm
        # 法向量 (左: [-dy, dx], 右: [dy, -dx])
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(p1 + path_width * normal)
        offsets_right.append(p1 - path_width * normal)
    
    # 最后一个点也加上
    if len(waypoints) > 1:
        direction = waypoints[-1] - waypoints[-2]
        dir_unit = direction / np.linalg.norm(direction)
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(waypoints[-1] + path_width * normal)
        offsets_right.append(waypoints[-1] - path_width * normal)

    offsets_left = np.array(offsets_left)
    offsets_right = np.array(offsets_right)

    # ========== step2: 投影函数 ==========
    def project_points(points):
        num_points = points.shape[0]
        ego_points = np.zeros((num_points, 3))
        ego_points[:, :2] = points
        cam_translation = np.array([-0.1, 0, camera_height])
        R = np.array([
            [0, -1, 0],   
            [0, 0, -1],   
            [1, 0, 0]     
        ])
        cam_points = (R @ (ego_points - cam_translation).T).T
        valid_mask = cam_points[:, 2] > 0
        cam_points = cam_points[valid_mask]
        if cam_points.shape[0] == 0:
            return np.zeros((0,2))
        
        x = cam_points[:, 0] / cam_points[:, 2]
        y = cam_points[:, 1] / cam_points[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(r)
        theta_d = theta * (1 + k * theta**2)
        scaling = np.zeros_like(r)
        mask = r > 1e-8
        scaling[mask] = theta_d[mask] / r[mask]
        scaling[~mask] = 1.0
        x_distorted = x * scaling
        y_distorted = y * scaling
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        return np.column_stack((u, v))

    points_center = project_points(waypoints)
    points_left = project_points(offsets_left)
    points_right = project_points(offsets_right)

    # ========== step3: Create navigation-style path overlay ==========
    if len(points_left) > 1 and len(points_right) > 1:
        # Create smooth polygon for the path
        polygon = np.vstack([points_left, points_right[::-1]])
        polygon = polygon.astype(np.int32)
        
        # Create overlay with navigation green color
        overlay = img.copy()
        
        # Navigation green color (similar to the reference image)
        nav_green = (65, 121, 76)  # BGR format - forest green
        if color is not None:
            nav_green = color
        
        # Fill the path polygon
        cv2.fillPoly(overlay, [polygon], color=nav_green)
        
        # Blend with original image for semi-transparency
        alpha = 0.5  # Higher alpha for more prominent path
        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        
        # Add subtle border/outline to the path
        # cv2.polylines(img, [polygon], isClosed=True, color=(0, 100, 0), thickness=2)

    # ========== step4: Add center line for guidance ==========
    # Add the vehicle starting point at bottom center
    start_point = np.array([[W//2, H-1]])
    points_center = np.vstack((start_point, points_center))
    
    # Filter points within image bounds
    points_center = points_center[
        (points_center[:, 0] >= 0) & (points_center[:, 0] < W) &
        (points_center[:, 1] >= 0) & (points_center[:, 1] < H)
    ]
    
    # Draw center guidance line with dashed style
    # if points_center.shape[0] > 1:
    #     # Create a more prominent center line
    #     center_color = (34, 139, 34)  # White center line
    #     line_thickness = max(6, int(H/200))
        
    #     for pt_id in range(points_center.shape[0]-1):
    #         x1, y1 = int(np.round(points_center[pt_id][0])), int(np.round(points_center[pt_id][1]))
    #         x2, y2 = int(np.round(points_center[pt_id+1][0])), int(np.round(points_center[pt_id+1][1]))
            
    #         if all(0 <= x < W and 0 <= y < H for x, y in [(x1, y1), (x2, y2)]):
    #             # Draw dashed line for center guidance
    #             draw_dashed_line(img, (x1, y1), (x2, y2), center_color, line_thickness)

    # # ========== step5: Add distance markers (optional) ==========
    # # Add small markers at regular intervals along the path
    # if points_center.shape[0] > 2:
    #     marker_interval = max(1, len(points_center) // 5)  # 5 markers max
    #     for i in range(0, len(points_center), marker_interval):
    #         if i < len(points_center):
    #             x, y = int(points_center[i][0]), int(points_center[i][1])
    #             if 0 <= x < W and 0 <= y < H:
    #                 cv2.circle(img, (x, y), max(3, int(H/150)), (34, 109, 34), -1)

    # Resize to standard output size
    # img_resized = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
    
    if save_fig:
        cv2.imwrite(save_path, img)
    
    return img


def project_waypoints_to_fisheye_image_with_polygon_new(
        waypoints, intrinsic_params, image_path, save_path='projected_waypoints_fisheye.jpg', 
        camera_height=0.41, save_fig=False, color=None, base_alpha=0.15):
    """
    New fisheye projection function with gradient transparency and side bars.
    
    Parameters:
    - waypoints: Array of shape (N, 2) with x,y coordinates in meters
    - intrinsic_params: [f, cx, cy, k] where f is focal length, (cx,cy) is principal point, k is distortion coefficient
    - image_path: Path to fisheye image or numpy array
    - save_path: Output path for saving the result
    - camera_height: Height of camera above ground (meters)
    - save_fig: Whether to save the result image
    - color: Custom color for the path (BGR format)
    
    Returns:
    - img: Image with projected waypoints as navigation path
    """

    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints should be of shape (N, 2)")
    if len(intrinsic_params) != 4:
        raise ValueError("intrinsic_params should be [f, cx, cy, k]")
    
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    else:
        print(f'Unsupported image_path type: {type(image_path)}')

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    H, W = img.shape[:2]
    f, cx, cy, k = intrinsic_params

    # ========== step1: Generate left and right offset waypoints ==========
    # Make the path wider to match navigation style
    path_width = 0.22  # Increased width for navigation path
    offsets_left, offsets_right = [], []
    
    for i in range(len(waypoints)-1):
        p1, p2 = waypoints[i], waypoints[i+1]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        dir_unit = direction / norm
        # Normal vectors (left: [-dy, dx], right: [dy, -dx])
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(p1 + path_width * normal)
        offsets_right.append(p1 - path_width * normal)
    
    # Add the last point as well
    if len(waypoints) > 1:
        direction = waypoints[-1] - waypoints[-2]
        dir_unit = direction / np.linalg.norm(direction)
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(waypoints[-1] + path_width * normal)
        offsets_right.append(waypoints[-1] - path_width * normal)

    offsets_left = np.array(offsets_left)
    offsets_right = np.array(offsets_right)

    # ========== step2: Improved fisheye projection function ==========
    def project_points_fisheye(points):
        num_points = points.shape[0]
        ego_points = np.zeros((num_points, 3))
        ego_points[:, :2] = points
        cam_translation = np.array([-0.1, 0, camera_height])
        R = np.array([
            [0, -1, 0],   
            [0, 0, -1],   
            [1, 0, 0]     
        ])
        cam_points = (R @ (ego_points - cam_translation).T).T
        valid_mask = cam_points[:, 2] > 0
        cam_points = cam_points[valid_mask]
        if cam_points.shape[0] == 0:
            return np.zeros((0,2))
        
        # Normalize to unit sphere (perspective projection)
        x = cam_points[:, 0] / cam_points[:, 2]
        y = cam_points[:, 1] / cam_points[:, 2]
        
        # Calculate distance from optical center
        r = np.sqrt(x**2 + y**2)
        
        # Fisheye projection: theta = atan(r)
        theta = np.arctan(r)
        
        # Apply fisheye distortion model
        theta_d = theta * (1 + k * theta**2)
        
        # Handle division by zero for points at optical center
        scaling = np.ones_like(r)
        mask = r > 1e-8
        scaling[mask] = theta_d[mask] / r[mask]
        
        # Apply scaling to get distorted coordinates
        x_distorted = x * scaling
        y_distorted = y * scaling
        
        # Project to image plane using focal length and principal point
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        
        return np.column_stack((u, v))

    points_center = project_points_fisheye(waypoints)
    points_left = project_points_fisheye(offsets_left)
    points_right = project_points_fisheye(offsets_right)

    # ========== step3: Create navigation path with gradient transparency ==========
    if len(points_left) > 1 and len(points_right) > 1 and waypoints[-1, 0] > 1.:
        # Create polygon for the path
        polygon = np.vstack([points_left, points_right[::-1]])
        polygon = polygon.astype(np.int32)
        
        # Navigation green color (similar to the reference image)
        nav_green = (65, 121, 76)  # BGR format - forest green
        if color is not None:
            nav_green = color
        
        # Create mask for the path region
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        
        # Create gradient mask (stronger near bottom, weaker towards top)
        gradient_mask = np.zeros((H, W), dtype=np.float32)
        
        # For each pixel in the mask, calculate gradient based on distance along path
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) > 0:
            # Calculate normalized distance from bottom (1.0 at bottom, 0.0 at top)
            # Using y-coordinate for simple vertical gradient
            normalized_y = y_coords / H
            
            # Create gradient that's stronger at bottom (closer to camera)
            # Use exponential or power function for smooth gradient
            alpha_values = base_alpha + 0.45 * (normalized_y ** 1.5)  # Range from 0.15 to 0.6
            
            # Apply gradient to mask
            gradient_mask[y_coords, x_coords] = alpha_values
        
        # Apply the navigation path with gradient transparency
        overlay = img.copy()
        
        # Apply color to overlay where mask is active
        overlay[mask > 0] = nav_green
        
        # Blend with gradient transparency
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - gradient_mask) + overlay[:, :, c] * gradient_mask
        
        # ========== step4: Add narrow dark bars on left and right edges ==========
        # Calculate positions for edge bars based on the polygon bounds
        if len(points_left) > 0 and len(points_right) > 0 and waypoints[-1, 0] > 2.:
            # Create slightly offset lines for the edge bars
            bar_width = 5  # Very narrow bars (2-3 pixels)
            bar_color = (int(nav_green[0] * 0.7), int(nav_green[1] * 0.7), int(nav_green[2] * 0.7))  # Darker green/gray color
            
            # Left edge bar - follow the left boundary
            left_points = points_left.astype(np.int32)
            # Filter valid points
            valid_left = left_points[
                (left_points[:, 0] >= 0) & (left_points[:, 0] < W) &
                (left_points[:, 1] >= 0) & (left_points[:, 1] < H)
            ]
            
            if len(valid_left) > 1:
                # Draw polyline for left edge
                cv2.polylines(img, [valid_left], False, bar_color, bar_width, cv2.LINE_AA)
            
            # Right edge bar - follow the right boundary
            right_points = points_right.astype(np.int32)
            # Filter valid points
            valid_right = right_points[
                (right_points[:, 0] >= 0) & (right_points[:, 0] < W) &
                (right_points[:, 1] >= 0) & (right_points[:, 1] < H)
            ]
            
            if len(valid_right) > 1:
                # Draw polyline for right edge
                cv2.polylines(img, [valid_right], False, bar_color, bar_width, cv2.LINE_AA)
    
    # ========== step5: Optional - Add subtle center guidance ==========
    # Add the vehicle starting point at bottom center
    start_point = np.array([[W//2, H-1]])
    points_center = np.vstack((start_point, points_center))
    
    # Filter points within image bounds
    points_center = points_center[
        (points_center[:, 0] >= 0) & (points_center[:, 0] < W) &
        (points_center[:, 1] >= 0) & (points_center[:, 1] < H)
    ]
    
    # Optional: Add very subtle center line with gradient opacity
    # print(waypoints, waypoints.shape)
    if points_center.shape[0] > 1 and waypoints[-1, 0] > 2.:
        for i in range(len(points_center) - 1):
            pt1 = tuple(points_center[i].astype(int))
            pt2 = tuple(points_center[i+1].astype(int))
            
            # Calculate opacity based on y-position (stronger at bottom)
            y_avg = (pt1[1] + pt2[1]) / 2
            opacity = 0.1 + 0.2 * (y_avg / H)  # Very subtle, 0.1 to 0.3 opacity
            
            # Draw on overlay and blend
            overlay = img.copy()
            cv2.line(overlay, pt1, pt2, (int(nav_green[0] * 0.8), int(nav_green[1] * 0.8), int(nav_green[2] * 0.8)), 2, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, opacity, img, 1-opacity, 0)
    
    if save_fig:
        cv2.imwrite(save_path, img)
    
    return img

if __name__ == "__main__":
    map_type = 'X'
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="all", choices=["lidar", 'all'])
    parser.add_argument("--density_obj", type=float, default=0.7)
    parser.add_argument('--config', type=str, default='/home/hollis/projects/S2E/e2enav/configs/exp/s1_web_goalfree_vision_policy_64anchor_5s.yaml')
    parser.add_argument('--ckpt', type=str, default='/home/hollis/projects/S2E/last-v1.ckpt')
    args = parser.parse_args()

    config = dict(
        crswalk_density=1,
        object_density=args.density_obj,
        walk_on_all_regions=False,
        use_render=True,
        map=map_type,
        manual_control=False,
        default_expert=False,
        drivable_area_extension=55,
        height_scale=1,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        
        agent_type='coco', #['coco', 'wheelchair']
        window_size=(1200, 900),
    )

    model = load_model()
    print(f'Successfully loaded model from {args.ckpt}')
    total_params = sum(p.numel() for p in model.parameters()) / 1024 / 1024
    print(f'Total parameters: {total_params:.2f}M')
    model = model.to('cuda').eval()

    if args.observation == "all":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(
                    rgb_camera=(RGBCamera, 1920, 1080),
                    depth_camera=(DepthCamera, 640, 640),
                    semantic_camera=(SemanticCamera, 640, 640),
                ),
                agent_observation=ThreeSourceMixObservation,
                interface_panel=[]
            )
        )

    env = SidewalkCleanMetaUrbanEnv(config)
    o, _ = env.reset(seed=48)
    image_list = []
    # env.engine.toggleDebug()
    logger = get_logger()
    logger.info("Please make sure that you have pulled all assets for the simulator, or the results may not be as expected.")

    action = [0.0, 0.0]

    try:
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):

            o, r, tm, tc, info = env.step(action)  ### reset; get next -> empty -> have multiple end points
            # print(o['image'][..., -1].shape, o['image'][..., -1].max(), o['image'][..., -1].min())
            image = o['image'][..., -1]
            image = (image * 255).astype(np.uint8)[..., ::-1] # RGB
            if len(image_list) == 0:
                image_list = [image] * 16
            else:
                image_list.append(image)
                image_list = image_list[-16:]

            image_observation = np.stack(image_list, axis=0)
            image_observation = image_observation.transpose(0, 3, 1, 2)
            image_observation = torch.from_numpy(image_observation).to('cuda').float()
            image_observation = image_observation / 255.0
            image_observation = image_observation.float()
            image_observation = torch.nn.functional.interpolate(image_observation, size=(288, 512), mode='bilinear', align_corners=False)
            image_observation = image_observation.unsqueeze(0)
            observation = {
                'observation': image_observation,
                'past_mask': torch.ones([1, 15], device='cuda').bool(),
                'future_mask': torch.ones([1, 16], device='cuda').bool(),
                'metric_spacing': torch.ones([1, ], device='cuda').float() * 0.25,
            }

            trajectory_action = model.model.forward_test(observation, infer_only=True)['trajectory'][0] * 0.25
            trajectory_action = trajectory_action.cpu().detach().numpy()[..., :2]
            intrinsic_params = [790, 1920 /2, 1080 /2, 0.0]
            image_path = project_waypoints_to_fisheye_image_with_polygon_new(
                trajectory_action, intrinsic_params, image, 'pred_3_save_path', save_fig=False, color=(255, 153, 51), base_alpha=0.35
            )[..., ::-1]

            action = [trajectory_action[5, 0] / 5.0, np.arctan2(trajectory_action[5, 1], trajectory_action[5, 0]) / (np.pi / 6)]

            cv2.imshow('image', image_path[..., ::-1])
            cv2.waitKey(1)

            if (tm or tc):
                env.reset(((env.current_seed + 1) % config['num_scenarios']) + env.engine.global_config['start_seed'])
                image_list = []
                image = o['image'][..., -1]
                image = (image * 255).astype(np.uint8)[..., ::-1] # RGB
                if len(image_list) == 0:
                    image_list = [image] * 16
                else:
                    image_list.append(image)
                    image_list = image_list[-16:]
                action = [0.0, 0.0]

    finally:
        env.close()
