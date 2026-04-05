"""
Manual download list for building models from Poly Haven and other sources.
All models are CC0 licensed (free to use commercially).

Instructions:
1. Review the links below
2. Download GLB files
3. Save to: metaurban/assets/models/scenes/{scene_type}/
"""

DOWNLOAD_MANIFEST = {
    "commercial": {
        "description": "High-interaction commercial buildings",
        "models": [
            {
                "name": "Modern Glass Building",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/modern_glass_building",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/modern_glass_building/model.glb",
                "license": "CC0"
            },
            {
                "name": "City Building 01",
                "source": "Sketchfab (manual download)",
                "url": "https://sketchfab.com/models/abc123def456",
                "note": "Search 'city building free' on Sketchfab, download GLB"
            },
            {
                "name": "Office Block",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/office_building",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/office_building/model.glb",
                "license": "CC0"
            },
            {
                "name": "Shopping Mall",
                "source": "3D Model Sites",
                "note": "CGTrader Free or Sketchfab free section"
            }
        ]
    },
    "commute": {
        "description": "Professional office/commute buildings",
        "models": [
            {
                "name": "Corporate Office Tower",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/office_tower",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/office_tower/model.glb",
                "license": "CC0"
            },
            {
                "name": "Business Building",
                "source": "Sketchfab (manual download)",
                "url": "https://sketchfab.com/search?q=business+building&license=cc0",
                "note": "Search and download CC0 licensed models"
            },
            {
                "name": "Industrial Building",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/industrial_building",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/industrial_building/model.glb",
                "license": "CC0"
            },
            {
                "name": "Modern Office Complex",
                "source": "Sketchfab",
                "url": "https://sketchfab.com/search?q=office+complex&license=cc0_pddl"
            }
        ]
    },
    "leisure": {
        "description": "Park and leisure area structures",
        "models": [
            {
                "name": "Pavilion Structure",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/pavilion",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/pavilion/model.glb",
                "license": "CC0"
            },
            {
                "name": "Park Bench Shelter",
                "source": "Sketchfab",
                "url": "https://sketchfab.com/search?q=park+shelter&license=cc0",
                "note": "Download CC0 licensed models"
            },
            {
                "name": "Gazebo",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/gazebo",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/gazebo/model.glb",
                "license": "CC0"
            },
            {
                "name": "Stone Monument",
                "source": "Sketchfab",
                "url": "https://sketchfab.com/search?q=monument&license=cc0"
            }
        ]
    },
    "constrained": {
        "description": "Old narrow alley buildings",
        "models": [
            {
                "name": "Historic European Building",
                "source": "Sketchfab",
                "url": "https://sketchfab.com/search?q=old+building+historic&license=cc0",
                "note": "Search and download free historic building models"
            },
            {
                "name": "Traditional Shophouse",
                "source": "Sketchfab",
                "url": "https://sketchfab.com/search?q=shophouse+traditional&license=cc0"
            },
            {
                "name": "Narrow Urban Structure",
                "source": "Poly Haven",
                "url": "https://polyhaven.com/a/narrow_building",
                "glb_url": "https://cdn.polyhaven.com/asset_files/models/narrow_building/model.glb",
                "license": "CC0"
            },
            {
                "name": "Vintage Alley Building",
                "source": "Sketchfab",
                "url": "https://sketchfab.com/search?q=vintage+alley&license=cc0"
            }
        ]
    }
}

if __name__ == "__main__":
    import json
    print(json.dumps(DOWNLOAD_MANIFEST, indent=2))
    
    print("\n" + "="*70)
    print("BUILDING MODEL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    for scene_type, info in DOWNLOAD_MANIFEST.items():
        print(f"\n📦 {scene_type.upper()}: {info['description']}")
        print("-" * 70)
        
        for i, model in enumerate(info['models'], 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Source: {model['source']}")
            print(f"   License: {model.get('license', 'Check website')}")
            
            if 'glb_url' in model:
                print(f"   Direct Download: {model['glb_url']}")
            elif 'url' in model:
                print(f"   Download from: {model['url']}")
            
            if 'note' in model:
                print(f"   Note: {model['note']}")
    
    print("\n" + "="*70)
    print("Steps:")
    print("1. Create directories: metaurban/assets/models/scenes/{scene_type}/")
    print("2. Download GLB files and place in appropriate directory")
    print("3. Generate metadata JSON files (see next script)")
    print("="*70)
