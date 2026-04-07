"""
SCENE TYPES FEATURE IMPLEMENTATION
===================================

This document describes the scene diversification feature added to the
social offline learning pipeline.

## What is Scene Diversification?

The environment now supports 4 distinct scene types that simulate different
urban contexts with different social dynamics:

1. **Commercial** (XSXSX pattern)
   - High-interaction commercial buildings
   - Busy shopping districts, malls, offices
   - Pedestrians expected to be attentive but navigating with other shopping tasks

2. **Commute** (SCSCS pattern)
   - Professional office/business buildings
   - CBD commute corridors, business parks
   - Pedestrians focused on efficient transit between workplaces

3. **Leisure** (SCSX pattern)
   - Parks and leisure structures
   - Pavilions, gazebos, shelters, monuments
   - Pedestrians expecting relaxed movement in unstructured routes

4. **Constrained** (X pattern)
   - Historic and narrow buildings  
   - Old narrow alleys, traditional structures
   - Constraints force more careful navigation behavior

## Architecture

### Core Components

1. **SceneBuilder** (`metaurban/manager/scene_builder.py`)
   - Manages scene-specific building asset pools
   - Loads building metadata and GLB files for each scene type
   - Provides utilities to select random buildings

2. **SocialScenarioManager** (extended)
   - Receives scene_type parameter in constructor
   - Logs scene type during reset for verification

3. **SocialDynamicMetaUrbanEnv** (extended)
   - Accepts `scene_type` configuration parameter
   - Passes scene_type to SocialScenarioManager

4. **Collection Script** (`collect_dataset.py`)
   - Added `--scene-type` CLI argument
   - Propagates scene_type to environment config

### Building Asset Structure

```
metaurban/assets/models/scenes/
├── commercial/
│   ├── modern_building_01.glb
│   ├── commercial_block_01.glb
│   ├── retail_building_01.glb
│   ├── office_tower_01.glb
│   ├── shopping_center_01.glb
│   └── manifest.json
├── commute/
│   ├── office_building_01.glb
│   ├── corporate_block_01.glb
│   ├── business_tower_01.glb
│   ├── industrial_building_01.glb
│   ├── office_complex_01.glb
│   └── manifest.json
├── leisure/
│   ├── pavilion_01.glb
│   ├── gazebo_01.glb
│   ├── park_shelter_01.glb
│   ├── bench_area_01.glb
│   ├── monument_01.glb
│   └── manifest.json
├── constrained/
│   ├── historic_building_01.glb
│   ├── narrow_building_01.glb
│   ├── traditional_house_01.glb
│   ├── old_warehouse_01.glb
│   ├── vintage_shophouse_01.glb
│   └── manifest.json
└── scenes_manifest.json
```

## Usage Examples

### Using SocialDynamicMetaUrbanEnv with scene types

```python
from metaurban.envs.social_dynamic_env import SocialDynamicMetaUrbanEnv

# Create environment with commercial scene
config = {
    "scene_type": "commercial",
    "use_render": False,
    "map": "XSXSX",  # Optional: can be auto-selected from SceneBuilder
    "horizon": 1000,
}
env = SocialDynamicMetaUrbanEnv(config)
obs, info = env.reset()
```

### Using SceneBuilder directly

```python
from metaurban.manager.scene_builder import SceneBuilder

# Load commercial scene
builder = SceneBuilder(scene_type="commercial")

# Get random building from the pool
building = builder.get_random_building()
print(f"Building: {building['name']}, scale: {building['scale']}")

# Get recommended map pattern for this scene
map_pattern = builder.get_map_pattern()
print(f"Recommended map pattern: {map_pattern}")

# Get all buildings in the scene
all_buildings = builder.get_all_buildings()
print(f"Total buildings: {len(all_buildings)}")
```

### Data Collection with Scene Types

```bash
# Collect 100 episodes in commercial scene
python metaurban/social_reward/collect_dataset.py \
    --num-episodes 100 \
    --scene-type commercial \
    --out-dir dataset/commercial \
    --env-mode social \
    --crossing-ped-num 8 \
    --vulnerable-ped-num 4 \
    --group-ped-pair-num 3

# Collect 100 episodes in leisure scene
python metaurban/social_reward/collect_dataset.py \
    --num-episodes 100 \
    --scene-type leisure \
    --out-dir dataset/leisure \
    --env-mode social

# Multi-scene collection (4 scene types × 50 episodes = 200 total)
for scene in commercial commute leisure constrained; do
    python metaurban/social_reward/collect_dataset.py \
        --num-episodes 50 \
        --scene-type $scene \
        --out-dir dataset/multi_scene \
        --env-mode social
done
```

## Implementation Details

### Scene Type Parameter Propagation

1. **CLI** → `--scene-type` argument in collect_dataset.py
2. **Config** → Added to env_config dict
3. **Environment** → SocialDynamicMetaUrbanEnv reads from config
4. **Manager** → SocialScenarioManager receives in __init__
5. **Logger** → Scene type logged during reset for verification

### Building Pool Loading

When building pools are needed (future enhancement):

1. SceneBuilder reads `metaurban/assets/models/scenes/{scene_type}/manifest.json`
2. Manifest contains metadata for all buildings in the pool
3. Each building has: name, file, scale, description
4. Manager can query SceneBuilder for random or specific buildings

### Current Limitations

- Building models are dummy GLB files (spheres scaled to different sizes)
- Real building geometry needs to be sourced from external models
- Building loading in the actual scene geometry is not yet integrated
- Currently all scenes use the same randomly-placed buildings from the default pool

### Future Enhancements

1. **Actual Building Geometry**
   - Replace dummy GLB files with real building models
   - Consider CC-licensed assets or procedural generation
   - Add more variety within each scene type

2. **Scene-Aware Rendering**
   - Modify asset loader to use scene-specific building pools
   - Integrate building selection into sidewalk_manager
   - Ensure visual differentiation is perceptible in output images

3. **Scene-Aware Rewards**
   - Use VLLM to label differences between scene types
   - Train separate reward functions per scene
   - Or train unified reward with scene embedding

## Verification

Test the implementation with:

```bash
# Verify scene builder and environment integration
python verify_scene_types.py
```

Expected output should show:
- All 4 scene types loading 5 building models each
- Map patterns correctly associated with each scene
- Environment successfully creating with each scene type
- Social role histogram showing role distribution

## Files Changed

### New Files
- `metaurban/manager/scene_builder.py` - Scene asset management
- `verify_scene_types.py` - Verification script
- `scripts/create_dummy_buildings.py` - Dummy model generation

### Modified Files
- `metaurban/envs/social_dynamic_env.py` - Added scene_type config param
- `metaurban/manager/social_scenario_manager.py` - Added scene_type param
- `metaurban/social_reward/collect_dataset.py` - Added --scene-type CLI arg
- `.gitignore` - Added test script exclusions

## Testing Results

✓ SceneBuilder loaded all 4 scene types
✓ Each scene has 5 building models
✓ Environment creation with scenes successful
✓ Social role histogram functioning
✓ Scene type logged in manager.reset()

## Next Steps

1. Integrate actual building models (real GLB files)
2. Modify sidewalk_manager to use scene-specific building pools
3. Ensure visual differentiation in rendered output
4. Collect multi-scene dataset and train VLLM labeler
5. Measure if VLLM can distinguish between scene types
"""

# This is a pseudo-file for documentation purposes
# Place this content in documentation/SCENE_TYPES_FEATURE.md
