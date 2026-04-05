"""
SceneBuilder for scene-specific building asset management.

This module provides utilities to load buildings from scene-specific asset pools
based on the configured scene type (commercial, commute, leisure, constrained).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class SceneBuilder:
    """Manages scene-specific building asset pools."""

    SCENE_TYPES = ["commercial", "commute", "leisure", "constrained"]
    
    # Map from scene type to ideal map sequences
    SCENE_MAP_PATTERNS = {
        "commercial": "XSXSX",          # High-interaction commercial
        "commute": "SCSCS",              # Professional/office
        "leisure": "SCSX",               # Parks and leisure
        "constrained": "X",              # Old narrow alleys
    }

    def __init__(self, scene_type: str = "commercial"):
        """
        Initialize the scene builder.

        Args:
            scene_type: One of ['commercial', 'commute', 'leisure', 'constrained']
        """
        if scene_type not in self.SCENE_TYPES:
            logger.warning(
                f"Unknown scene_type '{scene_type}', defaulting to 'commercial'. "
                f"Valid options: {self.SCENE_TYPES}"
            )
            scene_type = "commercial"

        self.scene_type = scene_type
        self.assets_dir = self._get_assets_dir()
        self.building_pool = self._load_building_pool()

    @staticmethod
    def _get_assets_dir() -> Path:
        """Get the path to scene-specific asset pools."""
        # Relative to this file: metaurban/manager/scene_builder.py
        # -> metaurban/assets/models/scenes/
        module_dir = Path(__file__).parent.parent
        assets_dir = module_dir / "assets" / "models" / "scenes"
        return assets_dir

    def _load_building_pool(self) -> List[Dict]:
        """Load building metadata for the current scene type."""
        manifest_path = self.assets_dir / self.scene_type / "manifest.json"
        
        if not manifest_path.exists():
            logger.warning(
                f"Building manifest not found: {manifest_path}\n"
                f"Scene type '{self.scene_type}' will use default building pool."
            )
            return []

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                return manifest.get('models', [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse manifest {manifest_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading building pool for '{self.scene_type}': {e}")
            return []

    def get_random_building(self, rng=None):
        """
        Get a random building from the scene-specific pool.

        Args:
            rng: (optional) numpy random generator; if None, uses Python's random

        Returns:
            Dict with keys: name, file, scale, description
        """
        if not self.building_pool:
            return None

        if rng is not None:
            import numpy as np
            idx = rng.integers(0, len(self.building_pool))
        else:
            import random
            idx = random.randint(0, len(self.building_pool) - 1)

        return self.building_pool[idx]

    def get_building_by_name(self, name: str) -> Optional[Dict]:
        """Get a specific building by name."""
        for building in self.building_pool:
            if building['name'] == name:
                return building
        return None

    def get_all_buildings(self) -> List[Dict]:
        """Get all buildings in the current scene pool."""
        return self.building_pool

    def get_building_glb_path(self, building_name: str) -> Optional[str]:
        """
        Get the full GLB file path for a building.

        Args:
            building_name: Name of the building (without .glb extension)

        Returns:
            Absolute path to the GLB file, or None if not found
        """
        building = self.get_building_by_name(building_name)
        if not building:
            return None

        glb_path = (
            self.assets_dir
            / self.scene_type
            / building['file']
        )

        if glb_path.exists():
            return str(glb_path)

        logger.warning(f"Building file not found: {glb_path}")
        return None

    def get_map_pattern(self) -> str:
        """Get the recommended map block pattern for this scene type."""
        return self.SCENE_MAP_PATTERNS.get(self.scene_type, "XSXSX")

    @classmethod
    def list_available_scenes(cls) -> List[str]:
        """List all available scene types."""
        return cls.SCENE_TYPES.copy()

    @classmethod
    def get_scenes_manifest(cls) -> Dict:
        """Get the global scenes manifest."""
        manifest_path = cls._get_assets_dir() / "scenes_manifest.json"
        
        if not manifest_path.exists():
            return {}

        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load scenes manifest: {e}")
            return {}

    def __repr__(self) -> str:
        return (
            f"SceneBuilder(scene_type='{self.scene_type}', "
            f"buildings={len(self.building_pool)}, "
            f"map_pattern='{self.get_map_pattern()}')"
        )
