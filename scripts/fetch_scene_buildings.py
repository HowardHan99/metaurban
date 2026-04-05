#!/usr/bin/env python3
"""
Fetch building models from Sketchfab API and organize by scene type.

Usage:
    python fetch_scene_buildings.py --download  # Download models
    python fetch_scene_buildings.py --list-only # Just list URLs
"""

import argparse
import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin

# Sketchfab API endpoint
SKETCHFAB_API = "https://api.sketchfab.com/v3"

# Scene type to search keywords mapping
SCENE_CONFIGS = {
    "commercial": {
        "keywords": ["office building", "modern building", "high-rise", "retail shop"],
        "count": 5,
        "description": "High-interaction commercial area"
    },
    "commute": {
        "keywords": ["office building", "corporate", "business building", "modern office"],
        "count": 5,
        "description": "Directional commute corridor"
    },
    "leisure": {
        "keywords": ["pavilion", "gazebo", "park structure", "tree house", "low-rise"],
        "count": 5,
        "description": "Open-space leisure area"
    },
    "constrained": {
        "keywords": ["old building", "historic building", "alley", "vintage", "narrow"],
        "count": 5,
        "description": "Narrow-constrained alley"
    }
}

class SketchfabFetcher:
    def __init__(self, output_dir: str = "metaurban/assets/models/scenes"):
        self.output_dir = Path(output_dir)
        self.models_by_scene = {}
        
    def search_models(self, query: str, count: int = 5) -> List[Dict]:
        """Search Sketchfab for models matching query."""
        params = {
            "q": query,
            "license": "free",  # Only free models
            "count": count,
            "sort_by": "-likeCount",  # Sort by popularity
        }
        
        try:
            response = requests.get(
                f"{SKETCHFAB_API}/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("results", []):
                model = {
                    "name": result.get("name", "Unknown"),
                    "uid": result.get("uid"),
                    "url": f"https://sketchfab.com/models/{result.get('uid')}",
                    "author": result.get("user", {}).get("username", "Unknown"),
                    "like_count": result.get("likeCount", 0),
                    "view_count": result.get("viewCount", 0),
                    "license": result.get("license", {}).get("label", "Unknown"),
                }
                
                # Check if downloadable
                if result.get("allowDownload", False):
                    model["downloadable"] = True
                    model["download_url"] = f"https://sketchfab.com/models/{result.get('uid')}/download"
                else:
                    model["downloadable"] = False
                
                results.append(model)
            
            return results
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching Sketchfab: {e}")
            return []
    
    def fetch_all_scenes(self):
        """Fetch models for all scene types."""
        print("=" * 70)
        print("Fetching building models from Sketchfab...")
        print("=" * 70)
        
        for scene_type, config in SCENE_CONFIGS.items():
            print(f"\n📦 Scene Type: {scene_type.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Searching keywords: {', '.join(config['keywords'])}")
            
            scene_models = []
            
            # Search with each keyword
            for keyword in config["keywords"]:
                print(f"   → Searching '{keyword}'...")
                models = self.search_models(keyword, count=2)
                scene_models.extend(models)
            
            # Deduplicate by UID
            seen_uids = set()
            unique_models = []
            for model in scene_models:
                if model["uid"] not in seen_uids:
                    seen_uids.add(model["uid"])
                    unique_models.append(model)
            
            # Keep top N
            self.models_by_scene[scene_type] = unique_models[:config["count"]]
            print(f"   ✓ Found {len(self.models_by_scene[scene_type])} models")
    
    def print_summary(self):
        """Print summary of fetched models."""
        print("\n" + "=" * 70)
        print("SUMMARY: Available Models by Scene")
        print("=" * 70)
        
        for scene_type in SCENE_CONFIGS.keys():
            models = self.models_by_scene.get(scene_type, [])
            print(f"\n📍 {scene_type.upper()} ({len(models)} models):")
            
            for i, model in enumerate(models, 1):
                print(f"\n   {i}. {model['name']}")
                print(f"      By: {model['author']}")
                print(f"      URL: {model['url']}")
                print(f"      Likes: {model['like_count']} | Views: {model['view_count']}")
                print(f"      Downloadable: {'✓ Yes' if model['downloadable'] else '✗ No'}")
                print(f"      License: {model['license']}")
    
    def generate_download_script(self):
        """Generate shell script for manual download."""
        script_path = Path("scripts/download_buildings.sh")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Download building models from Sketchfab\n\n")
            
            for scene_type in SCENE_CONFIGS.keys():
                models = self.models_by_scene.get(scene_type, [])
                f.write(f"\n# {scene_type.upper()}\n")
                f.write(f"mkdir -p metaurban/assets/models/scenes/{scene_type}\n\n")
                
                for model in models:
                    if model["downloadable"]:
                        f.write(f"# {model['name']}\n")
                        f.write(f"# URL: {model['url']}/download\n")
                        f.write(f"# Please manually download from URL above and save to:\n")
                        f.write(f"# metaurban/assets/models/scenes/{scene_type}/{model['uid']}.glb\n\n")
        
        print(f"\n✓ Download script generated: {script_path}")
        print("  Please manually visit the URLs and download GLB files.")
    
    def export_json_manifest(self):
        """Export model list as JSON manifest."""
        manifest = {
            "generated_at": str(Path.cwd()),
            "scenes": {}
        }
        
        for scene_type, models in self.models_by_scene.items():
            manifest["scenes"][scene_type] = [
                {
                    "name": m["name"],
                    "uid": m["uid"],
                    "author": m["author"],
                    "url": m["url"],
                    "downloadable": m["downloadable"],
                    "license": m["license"],
                }
                for m in models
            ]
        
        manifest_path = Path("metaurban/assets/models/scenes_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Manifest exported: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch building models from Sketchfab for scene types."
    )
    parser.add_argument("--list-only", action="store_true",
                        help="Only list URLs, don't download")
    parser.add_argument("--output-dir", type=str,
                        default="metaurban/assets/models/scenes",
                        help="Output directory for models")
    
    args = parser.parse_args()
    
    fetcher = SketchfabFetcher(output_dir=args.output_dir)
    
    # Fetch models
    fetcher.fetch_all_scenes()
    
    # Print results
    fetcher.print_summary()
    
    # Generate manifest
    fetcher.export_json_manifest()
    
    # Generate download script
    fetcher.generate_download_script()
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Review the manifest at: metaurban/assets/models/scenes_manifest.json")
    print("2. Manually download GLB files from Sketchfab URLs")
    print("3. Place files in: metaurban/assets/models/scenes/{scene_type}/")
    print("4. (Optional) Update scenes_manifest.json with local paths")
    print("=" * 70)


if __name__ == "__main__":
    main()
