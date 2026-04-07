# Social Navigation Features

Complete Feature Guide for Social Navigation Environment (MetaUrban env_change branch)

## 📋 Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Quick Start](#quick-start)
4. [Detailed Explanation](#detailed-explanation)
5. [Parameter Configuration](#parameter-configuration)
6. [Code Locations](#code-locations)

---

## Overview

This document describes three major new features for the MetaUrban Social Navigation Environment:

### 🎯 New Features

1. **Scene Type System** - Environment Diversification
   - 4 scene types: commercial, commute, leisure, constrained
   - Automatic switching between map generation patterns and building asset pools
    - Optional fallback to default building visuals when scene assets are not desired
    - `env-mode default` uses the social environment with default building visuals

2. **Improved Group Formation** - Natural Conversational Clustering
   - Concentric ring arrangement with member spacing of 1.35-2.45 meters
    - Group cluster center candidates are filtered to sidewalk regions
   - Automatic cluster release mechanism

3. **Vulnerable Pedestrians** - Social Diversity
   - Three subtypes: wheelchair, elderly, distracted
   - Distinct behavioral patterns and collision geometries

---

## Core Features

### Feature 1: Scene Type System

#### What is Scene Type?

Scene type (`scene_type`) controls two aspects:
1. **Map network generation pattern** - Road layout variations
2. **Building asset pool** - 3D models for different industries/purposes

#### Four Scene Types

| Scene Type | Map Pattern | Building Types | Use Case |
|-----------|------------|-----------------|----------|
| **commercial** | XSXSX (cross) | Shops, malls, office buildings | Commercial areas |
| **commute** | SCSCS (sequential) | Office towers, train stations, parking | Commute areas |
| **leisure** | SCSX (mixed) | Park pavilions, restaurants, cafes | Leisure areas |
| **constrained** | X (single) | Constrained-space buildings | Restricted environments |

#### Usage Examples

```bash
# Commercial scene
python metaurban/social_reward/collect_dataset.py \
  --env-mode social \
  --scene-type commercial \
  --num-episodes 10 \
  --horizon 500

# Leisure scene
python metaurban/social_reward/collect_dataset.py \
  --env-mode social \
  --scene-type leisure \
  --num-episodes 5
```

#### Asset Locations

- **Scene-specific buildings**: [metaurban/assets/models/scenes/](metaurban/assets/models/scenes/)
  - `commercial/` - Commercial buildings (modern_building_01.glb, etc.)
  - `commute/` - Commute buildings
  - `leisure/` - Leisure buildings
  - `constrained/` - Constrained buildings

- **Configuration manifests**: `models/scenes/{type}/manifest.json`

#### Building Asset Source

The building visual source is controlled by `scene_building_source`:

| Value | Behavior |
|-------|----------|
| `scene` | Use scene-specific building GLBs from `models/scenes/{scene_type}/` |
| `default` | Keep the original default building visuals from the base asset metadata |

Example:

```python
config = dict(
    env_mode='social',
    scene_type='commercial',
    scene_building_source='default',
)
```

The CLI alias `--env-mode default` is equivalent to using the social environment with `scene_building_source='default'`.

---

### Feature 2: Improved Group Formation

#### Group Lifecycle

```
Initialization Phase (typically 5-8 members per cluster)
     ↓
Clustering Phase (residing in formation, default 180±40 steps)
     ↓
Release Phase (gradually converting to normal pedestrians, independent walking)
     ↓
Completion Phase (dispersing to individual destinations)
```

#### Formation Structure: Concentric Rings

```
             ← Phase offset: each cluster has different orientation
         Inner Circle (≤6 members)
          •  •
         •    •    ← Radius 1.35m
          •  •

   Outer Circles (7-12 members)   ← Radius 1.35 + 0.55 = 1.9m
  •   •      •  •
 •       •        •
  •  •       •  •

All members face center, forming a "chat circle"
```

#### Key Parameters

| Parameter | Default | Description | CLI Flag |
|-----------|---------|-------------|----------|
| Cluster count | 3 | Number of clusters spawned simultaneously | `--group-cluster-num` |
| Member range | 5-8 | Members per cluster | `--group-cluster-size-min/max` |
| Member radius | 1.35m | Inner ring radius | config-only (`group_member_radius`) |
| Ring gap | 0.55m | Outer ring spacing | config-only (`group_member_ring_step`) |
| Cluster spacing | 3.8m | Minimum separation distance | config-only (`group_cluster_min_separation`) |
| Release mean | 180 | Average lifespan (steps) | `--group-release-steps-mean` |
| Release std dev | 40 | Lifespan variance | `--group-release-steps-std` |
| Minimum lifespan | 60 | Minimum residence steps | `--group-release-steps-min` |

#### Sidewalk Constraint Scope

- Group cluster centers are constrained to sidewalk polygons during cluster placement.
- Non-group pedestrians are not globally constrained by this patch and may still use the default walkable sampling behavior.
- Current default in social config is `pedestrian_sidewalk_only=False`.

#### Usage Examples

```bash
# Create 3 clusters with 7 members each, release after ~150 steps
python metaurban/social_reward/collect_dataset.py \
  --env-mode social \
  --scene-type commercial \
  --group-cluster-num 3 \
  --group-cluster-size-min 7 \
  --group-cluster-size-max 7 \
  --group-release-steps-mean 150 \
  --group-release-steps-std 30 \
  --use-render \
  --horizon 500

# Spawn near ego for easier visualization
python metaurban/social_reward/collect_dataset.py \
  --env-mode social \
  --group-spawn-near-ego \
  --group-spawn-min-radius 5 \
  --group-spawn-max-radius 10 \
  --num-episodes 5 \
  --use-render
```

#### Log Output Example

```
[INFO] Social role histogram: crossing=0 vulnerable=0 group=22 normal=20
[INFO] Released group cluster 0 with 6 pedestrians
[INFO] Released group cluster 1 with 8 pedestrians
```

---

### Feature 3: Vulnerable Pedestrians

#### Three Vulnerable Pedestrian Subtypes

| Subtype | Model Source | Collision Body | Behavioral Profile |
|---------|-------------|-----------------|-------------------|
| **wheelchair** | Dedicated wheelchair model | Radius 0.35m | Speed 0.45-0.70x, high yield (1.30-1.70x) |
| **elderly** | Standard pedestrian model | Radius 0.30m, mass 68kg | Speed 0.55-0.80x, medium yield (1.15-1.45x) |
| **distracted** | Standard pedestrian model | Radius 0.35m, mass 70kg | Speed 0.70-0.95x, low yield (0.90-1.10x), can pause |

#### Distribution Configuration

```bash
# Let the system automatically assign ratios
python metaurban/social_reward/collect_dataset.py \
  --env-mode social \
  --vulnerable-ped-num 8 \
  --vulnerable-elderly-ratio 0.6    # 60% elderly
  --vulnerable-distracted-ratio 0.4 # 40% distracted
  --spawn-elderly-num 4              # Hard-specify 4 ElderlyPedestrians
```

#### Behavioral Differences

- **Wheelchair**: Slow movement, higher tendency to yield to ego (1.30-1.70x yield radius)
- **Elderly**: Moderate speed and yield behavior
- **Distracted**: 
  - May randomly pause (2% probability/step, configurable)
  - Pause duration 16±5.6 steps
  - Less likely to yield (0.90-1.10x yield)

#### Output Example

```
[INFO] Vulnerable subtype histogram: wheelchair=1 elderly=4 distracted=3
```

---

## Quick Start

### Basic Environment Creation

```python
from metaurban.envs import SocialDynamicMetaUrbanEnv

# Commercial scene with groups
config = dict(
    env_mode='social',
    scene_type='commercial',
    group_cluster_num=3,
    vulnerable_ped_num=4,
    horizon=500,
    use_render=True,
)

env = SocialDynamicMetaUrbanEnv(config)
```

### Data Collection Example

```bash
# Complete example: combining all features
python metaurban/social_reward/collect_dataset.py \
  --num-episodes 20 \
  --env-mode social \
  --scene-type commercial \
  --horizon 500 \
  --crossing-ped-num 8 \
  --vulnerable-ped-num 6 \
  --group-cluster-num 4 \
  --group-cluster-size-min 5 \
  --group-cluster-size-max 8 \
  --group-release-enable \
  --group-release-steps-mean 180 \
  --group-spawn-near-ego \
  --use-render \
  --out-dir ./output/social_data
```

---

## Detailed Explanation

### 1. How Scene Type Works

#### Map Generation Pipeline

```
SocialDynamicMetaUrbanEnv._post_process_config()
    ↓
Read scene_type parameter
    ↓
Query SceneBuilder.get_map_pattern(scene_type)
    ↓
Obtain map pattern string (e.g., "XSXSX" → commercial)
    ↓
Set map_config["block_sequence"] = pattern directly
    ↓
Map generator creates corresponding road network layout
```

#### Building Asset Pool Loading Pipeline

```
SocialDynamicMetaUrbanEnv._setup_environment()
    ↓
SidewalkManager._apply_scene_building_pool()
    ↓
AssetManager loads manifest.json
    ↓
Randomly sample building GLBs from models/scenes/{scene_type}/
    ↓
Overwrite each Building's filename/scale attributes
    ↓
Rendering loads scene-specific building models
```

### 2. How Group Formation Works

#### Clustering Initialization (_init_roles phase)

```python
# Step 1: Calculate target cluster count
target_cluster_num = min(
    configured_num,
    available_pedestrians // min_cluster_size
)

# Step 2: Create clusters
for cluster_id in range(target_cluster_num):
    # Randomly select 5-8 members
    cluster_size = random(size_min, size_max)
    members = select_random_pedestrians(cluster_size)
    
    # Assign roles
    members[0] → ROLE_GROUP_LEADER
    members[1:] → ROLE_GROUP_FOLLOWER
    
    # Initialize cluster attributes
    cluster_center = leader_position
    cluster_phase = random(0, 2π)  # Random orientation offset
    release_counter = max(
        min_steps,
        round(normal(mean_steps, std_steps))
    )
```

#### Group Center Candidate Filtering (Sidewalk-only)

```python
def _pick_cluster_center_candidates(...):
    candidates = all_current_ped_positions()
    sidewalk_mask = build_sidewalk_only_mask()  # sidewalks + sidewalk buffers
    sidewalk_candidates = [p for p in candidates if is_point_on_mask(p, sidewalk_mask)]
    if sidewalk_candidates:
        candidates = sidewalk_candidates
    # then continue the existing template-based placement and min-separation checks
```

#### Per-Step Execution (after_step phase)

```python
# Step 1: Check release timers
if group_release_enable:
    for cluster_id in release_counter:
        release_counter[cluster_id] -= 1
        if release_counter[cluster_id] <= 0:
            _release_cluster(cluster_id)

# Step 2: Calculate member positions
for member_idx in traffic_humanoids:
    role = ped_role[member_idx]
    
    if role == ROLE_GROUP_LEADER or ROLE_GROUP_FOLLOWER:
        cluster_id = group_cluster_map[member_idx]
        
        # Calculate concentric ring position
        slot = group_member_slot[member_idx]
        offset = calculate_ring_offset(cluster_id, slot)
        actual_pos = cluster_center + offset
        target_pos = actual_pos
    else:
        # Non-group members use global ORCA trajectory
        target_pos = global_trajectory[member_idx]
    
    # Apply yield behavior and other logic
    ...
```

#### Release Phase (_release_cluster method)

```python
def _release_cluster(self, cluster_id):
    members = group_members[cluster_id]
    
    # Role conversion
    for member_idx in members:
        ped_role[member_idx] = ROLE_NORMAL
        traffic_humanoids[member_idx].social_role = ROLE_NORMAL
    
    # Clean cluster data
    group_cluster_map.remove(cluster_id)
    group_member_slot.remove(cluster_id)
    release_counter.remove(cluster_id)
    
    # Members transition to global trajectory system
    # Next step automatically follows ORCA-planned path to destination
```

### 3. How Vulnerable Pedestrians Work

#### Initialization Phase

```python
def _init_vulnerable_profiles(self):
    for idx, role in enumerate(ped_role):
        if role != ROLE_VULNERABLE:
            continue
        
        ped = traffic_humanoids[idx]
        cls_name = ped.__class__.__name__
        
        # Determine subtype and behavioral parameters based on class
        if "wheelchair" in cls_name:
            subtype = VUL_SUB_WHEELCHAIR
            speed_scale = uniform(0.45, 0.70)
            yield_scale = uniform(1.30, 1.70)
        
        elif "elderly" in cls_name:
            subtype = VUL_SUB_ELDERLY
            speed_scale = uniform(0.55, 0.80)
            yield_scale = uniform(1.15, 1.45)
        
        else:
            # Randomly assign elderly or distracted
            subtype = sample_subtype(elderly_ratio, distracted_ratio)
            
            if subtype == VUL_SUB_ELDERLY:
                speed_scale = uniform(0.55, 0.80)
                yield_scale = uniform(1.10, 1.40)
            else:  # distracted
                speed_scale = uniform(0.70, 0.95)
                yield_scale = uniform(0.90, 1.10)
        
        # Store parameters for later use
        vulnerable_subtype[idx] = subtype
        vulnerable_speed_scale[idx] = speed_scale
        vulnerable_yield_scale[idx] = yield_scale
```

#### Per-Step Behavior

```python
# Distracted pedestrians may pause
if subtype == VUL_SUB_DISTRACTED:
    if pause_counter[idx] > 0:
        pause_counter[idx] -= 1
        continue  # Don't move this step
    
    if random() < pause_probability:
        duration = round(normal(mean_pause_steps, std_pause_steps))
        pause_counter[idx] = duration
        continue

# Decide whether to yield based on yield radius
personal_yield_radius = base_yield_radius * yield_scale[idx]
if distance_to_ego < personal_yield_radius:
    continue  # Yield (stop moving)
```

---

## Parameter Configuration

### Environment Config Dictionary

```python
config = dict(
    # Base environment
    env_mode='social',              # Must be 'social'
    scene_type='commercial',        # commercial/commute/leisure/constrained
    horizon=500,                    # Steps per episode
    use_render=True,                # Whether to render
    
    # Pedestrian configuration
    spawn_human_num=30,             # Total pedestrian count
    spawn_wheelchairman_num=1,      # Wheelchair pedestrian count
    spawn_elderly_num=2,            # Elderly pedestrian count
    
    # Social role configuration
    crossing_ped_num=8,             # Crossing pedestrian count
    vulnerable_ped_num=4,           # Vulnerable pedestrian count
    group_cluster_num=3,            # Number of group clusters
    
    # Vulnerable pedestrian subtype ratios
    vulnerable_elderly_ratio=0.6,   # Elderly ratio (non-wheelchair)
    vulnerable_distracted_ratio=0.4,# Distracted ratio
    vulnerable_pause_prob=0.02,     # Pause probability per step
    vulnerable_pause_steps_mean=16, # Mean pause duration
    
    # Group formation parameters
    group_cluster_size_min=5,       # Minimum cluster size
    group_cluster_size_max=8,       # Maximum cluster size
    group_member_radius=1.35,       # Inner ring radius
    group_member_ring_step=0.55,    # Outer ring spacing
    group_cluster_min_separation=3.8, # Minimum cluster separation
    
    # Group spawn location
    group_spawn_near_ego=False,     # Whether to spawn near ego
    group_spawn_min_radius=5.0,     # Min ego-relative distance
    group_spawn_max_radius=10.0,    # Max ego-relative distance
    
    # Group release configuration
    group_release_enable=True,      # Whether to enable release
    group_release_steps_mean=180,   # Mean release duration
    group_release_steps_std=40,     # Release duration std dev
    group_release_steps_min=60,     # Minimum lifespan

    # Walkable-region scope
    pedestrian_sidewalk_only=False, # False by default; group center sidewalk constraint is handled separately
    pedestrian_allow_crosswalk=False,
)

env = SocialDynamicMetaUrbanEnv(config)
```

### CLI Parameters

```bash
python metaurban/social_reward/collect_dataset.py \
  # Scene configuration
  --scene-type {commercial,commute,leisure,constrained} \
  --env-mode social \
  
  # Pedestrian configuration
  --spawn-human-num 30 \
  --spawn-wheelchairman-num 1 \
  --spawn-elderly-num 2 \
  
  # Social roles
  --crossing-ped-num 8 \
  --vulnerable-ped-num 4 \
  --vulnerable-elderly-ratio 0.6 \
  --vulnerable-distracted-ratio 0.4 \
  --vulnerable-pause-prob 0.02 \
  --vulnerable-pause-steps-mean 16 \
  
  # Group parameters
  --group-cluster-num 3 \
  --group-cluster-size-min 5 \
  --group-cluster-size-max 8 \
  --group-spawn-near-ego \
  --group-spawn-min-radius 5 \
  --group-spawn-max-radius 10 \
  --group-release-enable \
  --group-release-steps-mean 180 \
  --group-release-steps-std 40 \
  --group-release-steps-min 60 \
  
  # Other
  --num-episodes 10 \
  --horizon 500 \
  --use-render \
  --out-dir ./output
```

---

## Code Locations

### Core Implementation Files

| Module | Location | Purpose |
|--------|----------|---------|
| **SocialDynamicEnv** | [metaurban/envs/social_dynamic_env.py](metaurban/envs/social_dynamic_env.py) | Main environment class, scene_type integration |
| **SocialScenarioManager** | [metaurban/manager/social_scenario_manager.py](metaurban/manager/social_scenario_manager.py) | Role assignment, group formation, vulnerable behavior |
| **SidewalkManager** | [metaurban/manager/sidewalk_manager.py](metaurban/manager/sidewalk_manager.py) | Scene-specific building asset pools |
| **Agent Types** | [metaurban/component/agents/pedestrian/pedestrian_type.py](metaurban/component/agents/pedestrian/pedestrian_type.py) | WheelchairPedestrian, ElderlyPedestrian |
| **Data Collection Script** | [metaurban/social_reward/collect_dataset.py](metaurban/social_reward/collect_dataset.py) | CLI entry point and parameter parsing |

### Key Methods

| Method | File | Purpose |
|--------|------|---------|
| `_post_process_config()` | social_dynamic_env.py#L85 | Scene type → map pattern routing |
| `_apply_scene_building_pool()` | sidewalk_manager.py#L420 | Load scene-specific building GLBs |
| `_init_roles()` | social_scenario_manager.py#L199 | Role assignment and group initialization |
| `_group_slot_offset()` | social_scenario_manager.py#L320 | Calculate group member concentric ring positions |
| `_init_vulnerable_profiles()` | social_scenario_manager.py#L451 | Vulnerable pedestrian subtype assignment |
| `_release_cluster()` | social_scenario_manager.py#L422 | Cluster dissolution |
| `after_step()` | social_scenario_manager.py#L472 | Per-step group and vulnerable behavior updates |

### Asset Files

| Asset | Location | Description |
|-------|----------|-------------|
| **Building models** | [metaurban/assets/models/scenes/](metaurban/assets/models/scenes/) | Scene-specific GLB files |
| **Pedestrian models** | metaurban/assets/models/pedestrian/ | GLTF animation models |
| **Wheelchair models** | metaurban/assets/models/test/ | Wheelchair*.glb |
| **Scene manifests** | metaurban/assets/models/scenes/*/manifest.json | Building list per scene |

---

## Verification and Testing

### Verify Group Functionality

```bash
# Verify only group roles are assigned
python metaurban/social_reward/collect_dataset.py \
  --num-episodes 1 \
  --env-mode social \
  --crossing-ped-num 0 \
  --vulnerable-ped-num 0 \
  --group-cluster-num 4 \
  --horizon 400

# Expected output: Social role histogram: crossing=0 vulnerable=0 group=XX normal=YY
```

### Verify Group Sidewalk-Center Placement

```bash
python metaurban/social_reward/collect_dataset.py \
    --num-episodes 1 \
    --env-mode social \
    --scene-type commercial \
    --map C \
    --horizon 260 \
    --policy random \
    --object-density 0.05 \
    --crossing-ped-num 0 \
    --vulnerable-ped-num 0 \
    --spawn-human-num 48 \
    --group-cluster-num 4 \
    --group-cluster-size-min 5 \
    --group-cluster-size-max 8 \
    --group-spawn-near-ego \
    --group-spawn-min-radius 6 \
    --group-spawn-max-radius 11 \
    --group-release-enable \
    --group-release-steps-mean 40 \
    --group-release-steps-std 3 \
    --group-release-steps-min 30 \
    --ignore-success-done \
    --use-render \
    --out-dir /tmp/group_run_live_now
```

Recent successful log pattern:

```text
[INFO] Social role histogram: crossing=0 vulnerable=0 group=23 normal=27
[INFO] Released group cluster 2 with 5 pedestrians
[INFO] Released group cluster 3 with 6 pedestrians
[INFO] Released group cluster 0 with 7 pedestrians
[INFO] Released group cluster 1 with 5 pedestrians
[INFO] Episode ended! Scenario Index: 710 Reason: max step
```

### Verify Vulnerable Pedestrians

```bash
# Verify only vulnerable roles are assigned
python metaurban/social_reward/collect_dataset.py \
  --num-episodes 1 \
  --env-mode social \
  --crossing-ped-num 0 \
  --vulnerable-ped-num 8 \
  --group-cluster-num 0 \
  --spawn-elderly-num 4 \
  --spawn-human-num 30 \
  --horizon 300

# Expected output: Vulnerable subtype histogram: wheelchair=1 elderly=4 distracted=3
```

### Verify Scene Types

```bash
# Verify building differences across scenes
for scene in commercial commute leisure constrained; do
  python metaurban/social_reward/collect_dataset.py \
    --num-episodes 1 \
    --env-mode social \
    --scene-type $scene \
    --use-render \
    --horizon 300
done
```

### Verify Default Building Fallback

```bash
python metaurban/social_reward/collect_dataset.py \
    --num-episodes 1 \
    --env-mode social \
    --scene-type commercial \
    --scene-building-source default \
    --use-render \
    --horizon 300
```

If you prefer the alias form, use:

```bash
python metaurban/social_reward/collect_dataset.py \
    --num-episodes 1 \
    --env-mode default \
    --scene-type commercial \
    --use-render \
    --horizon 300
```

---

## FAQ

### Q: Why aren't group members moving?
**A:** This is by design. Group members maintain their formation until release, then move independently. Check the values of `group_release_enable` and `group_release_steps_mean`.

### Q: Why is vulnerable pedestrian speed so slow?
**A:** Vulnerable pedestrians are intentionally designed to move slowly, mimicking reduced mobility. Speed ranges are determined by the `speed_scale` of each subtype (0.45-0.95x).

### Q: Can I change the building models?
**A:** Yes. Modify the GLB filenames in `metaurban/assets/models/scenes/{scene_type}/manifest.json`.

### Q: Why are clusters overlapping?
**A:** If `group_cluster_min_separation < 3.8m`, overlaps may occur. Increase this parameter or reduce `group_cluster_num`.

---

## Related Resources

- 📖 [Main README](README.md)
- 🔧 [Environment API Documentation](documentation/source/system_design.ipynb)
- 📊 [Scene Type Details](documentation/SCENE_TYPES_FEATURE.md)
- 🏃 [Basic Usage Examples](metaurban/examples/basic_colab_usage.ipynb)

---

## Update History

- **2026-04-06**: Initial documentation created
  - Complete scene type system
  - Improved group formation algorithm
  - Vulnerable pedestrian subtypes
  - Comprehensive parameter documentation

- **2026-04-06**: Group-sidewalk placement scope update
    - Group cluster centers are now filtered by sidewalk-only mask
    - Global pedestrian sidewalk-only default reverted (`pedestrian_sidewalk_only=False`)
    - Added validation command and expected runtime log pattern

- **2026-04-06**: Default building fallback option
    - Added `scene_building_source=default` to keep default building visuals
    - Scene-specific building overwrites are skipped when fallback mode is enabled

