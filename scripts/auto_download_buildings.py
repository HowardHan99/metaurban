#!/usr/bin/env python3
"""
Automatically download building models for scene types from Poly Haven and other sources.

Usage:
    python auto_download_buildings.py                      # Download all models
    python auto_download_buildings.py --scene-type commercial  # Download specific scene
    python auto_download_buildings.py --dry-run             # Preview without downloading
"""

import argparse
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
import time

# ==================== Download Manifest ====================
# Models from Poly Haven (CC0 licensed, direct download URLs)
DOWNLOAD_SOURCES = {
    "commercial": {
        "description": "High-interaction commercial buildings",
        "models": [
            {
                "name": "modern_building_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/modern_glass_building/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "commercial_block_01", 
                "url": "https://cdn.polyhaven.com/asset_files/models/city_block_01/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "retail_building_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/storefront_01/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "office_tower_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/tower_modern/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "shopping_center_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/mall_structure/model.glb",
                "source": "Poly Haven"
            }
        ]
    },
    "commute": {
        "description": "Professional office/commute buildings",
        "models": [
            {
                "name": "office_building_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/office_tower/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "corporate_block_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/corporate_building/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "business_tower_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/business_tower/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "industrial_building_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/industrial_structure/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "office_complex_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/office_complex/model.glb",
                "source": "Poly Haven"
            }
        ]
    },
    "leisure": {
        "description": "Park and leisure area structures",
        "models": [
            {
                "name": "pavilion_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/pavilion/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "gazebo_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/gazebo/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "park_shelter_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/public_shelter/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "bench_area_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/seating_area/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "monument_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/monument/model.glb",
                "source": "Poly Haven"
            }
        ]
    },
    "constrained": {
        "description": "Old narrow alley buildings",
        "models": [
            {
                "name": "historic_building_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/historic_european/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "narrow_building_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/narrow_structure/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "traditional_house_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/traditional_building/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "old_warehouse_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/old_warehouse/model.glb",
                "source": "Poly Haven"
            },
            {
                "name": "vintage_shophouse_01",
                "url": "https://cdn.polyhaven.com/asset_files/models/vintage_shophouse/model.glb",
                "source": "Poly Haven"
            }
        ]
    }
}


class BuildingDownloader:
    def __init__(self, base_dir: str = "metaurban/assets/models/scenes"):
        self.base_dir = Path(base_dir)
        self.downloaded = []
        self.failed = []
        self.skipped = []
        
    def ensure_directories(self):
        """Create scene type directories."""
        for scene_type in DOWNLOAD_SOURCES.keys():
            scene_dir = self.base_dir / scene_type
            scene_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {scene_dir}")
    
    def download_file(self, url: str, output_path: Path, timeout: int = 30) -> bool:
        """Download a single GLB file."""
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"  ⚠️  Download failed: {e}")
            return False
    
    def download_scene_models(self, scene_type: str, dry_run: bool = False):
        """Download all models for a scene type."""
        if scene_type not in DOWNLOAD_SOURCES:
            print(f"❌ Unknown scene type: {scene_type}")
            return
        
        config = DOWNLOAD_SOURCES[scene_type]
        scene_dir = self.base_dir / scene_type
        
        print(f"\n📦 {scene_type.upper()}: {config['description']}")
        print("=" * 70)
        
        for i, model in enumerate(config["models"], 1):
            model_name = model["name"]
            output_file = scene_dir / f"{model_name}.glb"
            
            # Check if already exists
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"  {i}. {model_name} ({size_mb:.1f} MB) - ✓ Already exists")
                self.skipped.append(model_name)
                continue
            
            print(f"  {i}. {model_name}...")
            
            if dry_run:
                print(f"     [DRY RUN] Would download from: {model['url']}")
                continue
            
            # Download
            if self.download_file(model["url"], output_file):
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"     ✓ Downloaded ({size_mb:.1f} MB)")
                self.downloaded.append(model_name)
            else:
                self.failed.append(model_name)
            
            # Be polite to servers - small delay between downloads
            time.sleep(0.5)
    
    def generate_metadata(self, scene_type: str):
        """Generate JSON metadata for downloaded models."""
        scene_dir = self.base_dir / scene_type
        
        # Find all GLB files
        glb_files = list(scene_dir.glob("*.glb"))
        
        metadata = {
            "scene_type": scene_type,
            "description": DOWNLOAD_SOURCES[scene_type]["description"],
            "building_count": len(glb_files),
            "buildings": []
        }
        
        for glb_file in glb_files:
            size_bytes = glb_file.stat().st_size
            metadata["buildings"].append({
                "name": glb_file.stem,
                "path": str(glb_file.relative_to(self.base_dir.parent)),
                "size_bytes": size_bytes,
                "format": "glb"
            })
        
        # Write metadata
        metadata_file = scene_dir / "manifest.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  ✓ Metadata written: {metadata_file}")
        
        return metadata
    
    def download_all(self, dry_run: bool = False):
        """Download all scene models."""
        self.ensure_directories()
        
        print("\n" + "=" * 70)
        print("DOWNLOADING BUILDING MODELS")
        print("=" * 70)
        
        for scene_type in DOWNLOAD_SOURCES.keys():
            self.download_scene_models(scene_type, dry_run=dry_run)
        
        # Generate metadata for all scenes
        print("\n" + "=" * 70)
        print("GENERATING METADATA")
        print("=" * 70)
        
        all_metadata = {}
        for scene_type in DOWNLOAD_SOURCES.keys():
            meta = self.generate_metadata(scene_type)
            all_metadata[scene_type] = meta
        
        # Write global manifest
        global_manifest = self.base_dir / "scenes_manifest.json"
        with open(global_manifest, "w") as f:
            json.dump(all_metadata, f, indent=2)
        print(f"\n  ✓ Global manifest written: {global_manifest}")
        
        # Print summary
        self.print_summary(dry_run)
    
    def print_summary(self, dry_run: bool = False):
        """Print download summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if dry_run:
            print("DRY RUN MODE - No files were actually downloaded")
            total_models = sum(len(config["models"]) for config in DOWNLOAD_SOURCES.values())
            print(f"Would download: {total_models} models across 4 scene types")
        else:
            print(f"✓ Downloaded: {len(self.downloaded)} models")
            print(f"⚠️  Skipped (already exist): {len(self.skipped)} models")
            print(f"❌ Failed: {len(self.failed)} models")
            
            if self.failed:
                print("\nFailed downloads:")
                for name in self.failed:
                    print(f"  - {name}")
        
        print("\n" + "=" * 70)
        print("Next steps:")
        print("1. Review downloaded models in: metaurban/assets/models/scenes/")
        print("2. Models metadata saved to: metaurban/assets/models/scenes/scenes_manifest.json")
        print("3. Integrate with MetaUrban environment code")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Automatically download building models for scene types."
    )
    parser.add_argument("--scene-type", type=str, default=None,
                        choices=list(DOWNLOAD_SOURCES.keys()) + ["all"],
                        help="Download specific scene type or all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without actually downloading")
    parser.add_argument("--output-dir", type=str,
                        default="metaurban/assets/models/scenes",
                        help="Output directory for models")
    
    args = parser.parse_args()
    
    downloader = BuildingDownloader(base_dir=args.output_dir)
    
    if args.scene_type and args.scene_type != "all":
        # Single scene type
        downloader.ensure_directories()
        downloader.download_scene_models(args.scene_type, dry_run=args.dry_run)
        downloader.generate_metadata(args.scene_type)
        downloader.print_summary(args.dry_run)
    else:
        # All scenes
        downloader.download_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
