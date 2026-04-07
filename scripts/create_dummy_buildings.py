#!/usr/bin/env python3
"""
创建 dummy GLB 建筑模型用于开发和测试。
这些是占位符；生产环境中应使用真实的建筑资源。
"""

import os
import json
import struct
import math
from pathlib import Path

def create_simple_glb(filepath, scale=1.0):
    """
    创建一个最小的有效 GLB 文件（单个立方体）。
    
    GLB 文件结构：
    - Header (12 bytes)
    - JSON chunk (with padding)
    - Binary chunk (with padding)
    """
    
    # 简单的立方体顶点和索引
    vertices = [
        -0.5, -0.5, -0.5,   # 0
         0.5, -0.5, -0.5,   # 1
        -0.5,  0.5, -0.5,   # 2
         0.5,  0.5, -0.5,   # 3
        -0.5, -0.5,  0.5,   # 4
         0.5, -0.5,  0.5,   # 5
        -0.5,  0.5,  0.5,   # 6
         0.5,  0.5,  0.5,   # 7
    ]
    
    # 缩放顶点
    vertices = [v * scale for v in vertices]
    
    indices = [
        0, 1, 2, 1, 3, 2,  # front
        4, 6, 5, 5, 6, 7,  # back
        0, 2, 4, 4, 2, 6,  # left
        1, 5, 3, 3, 5, 7,  # right
        0, 4, 1, 1, 4, 5,  # bottom
        2, 3, 6, 6, 3, 7,  # top
    ]
    
    # 创建二进制数据
    vertex_bytes = struct.pack('<' + 'f' * len(vertices), *vertices)
    index_bytes = struct.pack('<' + 'H' * len(indices), *indices)
    
    # 构建 glTF JSON
    gltf_json = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{
            "mesh": 0,
            "translation": [0.0, 0.0, 0.0]
        }],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # float
                "count": len(vertices) // 3,
                "type": "VEC3",
                "min": [-0.5 * scale, -0.5 * scale, -0.5 * scale],
                "max": [0.5 * scale, 0.5 * scale, 0.5 * scale]
            },
            {
                "bufferView": 1,
                "componentType": 5123,  # uint16
                "count": len(indices),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes)},
            {"buffer": 0, "byteOffset": len(vertex_bytes), "byteLength": len(index_bytes)}
        ],
        "buffers": [{
            "byteLength": len(vertex_bytes) + len(index_bytes)
        }]
    }
    
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    
    # Pad JSON to 4-byte boundary
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b' ' * json_padding
    
    # Pad binary to 4-byte boundary
    binary_data = vertex_bytes + index_bytes
    binary_padding = (4 - (len(binary_data) % 4)) % 4
    binary_data += b'\x00' * binary_padding
    
    # GLB header
    glb_header = struct.pack('<III',
        0x46546C67,  # 'glTF' in little-endian
        2,           # version
        12 + 8 + len(json_bytes) + 8 + len(binary_data)  # total file length
    )
    
    # JSON chunk header
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    
    # Binary chunk header
    binary_chunk = struct.pack('<II', len(binary_data), 0x004E4942) + binary_data
    
    # Write GLB file
    with open(filepath, 'wb') as f:
        f.write(glb_header + json_chunk + binary_chunk)


def main():
    # 场景配置
    scene_configs = {
        'commercial': {
            'description': 'High-interaction commercial buildings',
            'models': [
                ('modern_building_01', 2.0),
                ('commercial_block_01', 2.5),
                ('retail_building_01', 1.8),
                ('office_tower_01', 3.0),
                ('shopping_center_01', 2.2),
            ]
        },
        'commute': {
            'description': 'Professional office/commute buildings',
            'models': [
                ('office_building_01', 2.0),
                ('corporate_block_01', 2.5),
                ('business_tower_01', 3.0),
                ('industrial_building_01', 1.5),
                ('office_complex_01', 2.8),
            ]
        },
        'leisure': {
            'description': 'Park and leisure area structures',
            'models': [
                ('pavilion_01', 1.2),
                ('gazebo_01', 1.0),
                ('park_shelter_01', 0.8),
                ('bench_area_01', 0.5),
                ('monument_01', 1.5),
            ]
        },
        'constrained': {
            'description': 'Old narrow alley buildings',
            'models': [
                ('historic_building_01', 1.8),
                ('narrow_building_01', 1.5),
                ('traditional_house_01', 1.2),
                ('old_warehouse_01', 2.0),
                ('vintage_shophouse_01', 1.6),
            ]
        }
    }
    
    base_dir = Path('metaurban/assets/models/scenes')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    scenes_manifest = {'scenes': {}}
    
    for scene_type, config in scene_configs.items():
        scene_dir = base_dir / scene_type
        scene_dir.mkdir(exist_ok=True)
        
        print(f"\n📦 {scene_type.upper()}: {config['description']}")
        print("=" * 70)
        
        scene_models = []
        
        for idx, (model_name, scale) in enumerate(config['models'], 1):
            glb_path = scene_dir / f"{model_name}.glb"
            
            try:
                create_simple_glb(str(glb_path), scale=scale)
                print(f"  ✓ Created: {model_name} (scale={scale})")
                
                scene_models.append({
                    'id': idx,
                    'name': model_name,
                    'file': f"{model_name}.glb",
                    'scale': scale,
                    'description': f"Dummy model for {scene_type} scene"
                })
            except Exception as e:
                print(f"  ❌ Failed: {model_name} - {e}")
        
        # 写入场景级别的 manifest
        manifest_path = scene_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump({
                'scene_type': scene_type,
                'description': config['description'],
                'models': scene_models,
                'total': len(scene_models)
            }, f, indent=2)
        
        print(f"  ✓ Manifest: {manifest_path}")
        scenes_manifest['scenes'][scene_type] = {
            'models': len(scene_models),
            'description': config['description']
        }
    
    # 写入全局 manifest
    global_manifest = base_dir / 'scenes_manifest.json'
    with open(global_manifest, 'w') as f:
        json.dump(scenes_manifest, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Dummy building models created successfully!")
    print(f"✓ Location: {base_dir}")
    print(f"✓ Total scenes: {len(scene_configs)}")
    print(f"✓ Total models: {sum(len(c['models']) for c in scene_configs.values())}")
    print("=" * 70)


if __name__ == '__main__':
    main()
