#!/usr/bin/env python3
"""Preview scene-specific GLB building assets with Panda3D.

Usage examples:
  python scripts/preview_scene_models.py --scene-type commercial
  python scripts/preview_scene_models.py --scene-type leisure --model gazebo_01
  python scripts/preview_scene_models.py --scene-type commute --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import registers glTF/.glb loader for Panda3D.
# Different environments expose different module names.
try:
    import panda3d_gltf  # noqa: F401
except Exception:
    try:
        import gltf  # type: ignore  # noqa: F401
    except Exception:
        # Keep running; some builds may already have .glb support registered.
        pass
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, LVector3


ROOT = Path(__file__).resolve().parents[1]
SCENES_ROOT = ROOT / "metaurban" / "assets" / "models" / "scenes"
VALID_SCENES = ["commercial", "commute", "leisure", "constrained"]


def load_scene_manifest(scene_type: str) -> dict:
    manifest_path = SCENES_ROOT / scene_type / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def resolve_model_path(scene_type: str, model_name: str | None) -> tuple[Path, float, str]:
    manifest = load_scene_manifest(scene_type)
    models = manifest.get("models", [])
    if not models:
        raise RuntimeError(f"No models found in {scene_type} manifest")

    target = None
    if model_name is None:
        target = models[0]
    else:
        for item in models:
            if item.get("name") == model_name:
                target = item
                break

    if target is None:
        names = ", ".join(m.get("name", "<unnamed>") for m in models)
        raise ValueError(f"Model '{model_name}' not found in scene '{scene_type}'. Available: {names}")

    path = SCENES_ROOT / scene_type / target["file"]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return path, float(target.get("scale", 1.0)), target.get("name", path.stem)


class PreviewApp(ShowBase):
    def __init__(self, model_path: Path, model_scale: float, title: str):
        super().__init__()
        self.disableMouse()

        self._title = title
        self._spin = True

        model = self.loader.loadModel(str(model_path))
        if model.isEmpty():
            raise RuntimeError(f"Failed to load model: {model_path}")

        model.reparentTo(self.render)
        model.setScale(model_scale)
        self._model = model

        # Center model near origin and fit camera by bounds.
        low, high = model.getTightBounds()
        center = (low + high) * 0.5
        extent = high - low
        radius = max(extent.length() * 0.5, 1.0)

        model.setPos(-center)
        self.camera.setPos(0, -max(8.0, radius * 3.0), max(2.0, radius * 1.2))
        self.camera.lookAt(0, 0, 0)

        self._setup_lighting()
        self.taskMgr.add(self._spin_task, "spin-model")

        self.accept("escape", self.userExit)
        self.accept("space", self._toggle_spin)

        print("=" * 70)
        print(f"Previewing: {title}")
        print(f"File: {model_path}")
        print("Controls: ESC quit, SPACE toggle rotation")
        print("=" * 70)

    def _setup_lighting(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.45, 0.45, 0.45, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key = DirectionalLight("key")
        key.setColor((0.85, 0.85, 0.85, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-35, -35, 0)
        self.render.setLight(key_np)

    def _toggle_spin(self) -> None:
        self._spin = not self._spin
        print(f"Auto rotation: {'ON' if self._spin else 'OFF'}")

    def _spin_task(self, task):
        if self._spin:
            self._model.setH(task.time * 20.0)
        return task.cont


def list_models(scene_type: str) -> int:
    manifest = load_scene_manifest(scene_type)
    print(f"Scene: {scene_type}")
    print(f"Description: {manifest.get('description', '')}")
    print("Models:")
    for item in manifest.get("models", []):
        print(f"  - {item.get('name')} ({item.get('file')}, scale={item.get('scale', 1.0)})")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview scene-specific GLB models.")
    parser.add_argument("--scene-type", default="commercial", choices=VALID_SCENES)
    parser.add_argument("--model", default=None, help="Model name from manifest, e.g. gazebo_01")
    parser.add_argument("--list", action="store_true", help="List available models for the scene")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        return list_models(args.scene_type)

    model_path, scale, resolved_name = resolve_model_path(args.scene_type, args.model)
    title = f"{args.scene_type}/{resolved_name}"

    app = PreviewApp(model_path=model_path, model_scale=scale, title=title)
    app.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
