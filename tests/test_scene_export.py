import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from depthwizard.reconstruction.dsm import depth_to_relative_elevation
from depthwizard.reconstruction.elevation_grid import build_elevation_grid
from depthwizard.reconstruction.mesh import build_terrain_mesh
from depthwizard.reconstruction.scene_export import (
    build_mesh_scene,
    build_scene,
    export_scene_json,
)


class TestBuildScene(unittest.TestCase):
    def test_build_scene_is_tagged_uncalibrated(self):
        depth = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        elevation = depth_to_relative_elevation(depth)
        scene = build_scene(
            elevation, rgb, depth_source="depth-anything-v2-small",
            depth_status="SUCCESS", max_points=100,
        )
        self.assertFalse(scene["calibrated"])
        self.assertEqual(scene["intrinsics_source"], "placeholder_uncalibrated")
        self.assertEqual(scene["depth_source"], "depth-anything-v2-small")
        self.assertEqual(scene["depth_status"], "SUCCESS")
        self.assertIn("elevation_convention", scene)
        self.assertGreater(scene["point_count"], 0)

    def test_export_scene_json_roundtrip(self):
        depth = np.random.default_rng(1).random((8, 8)).astype(np.float32)
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        elevation = depth_to_relative_elevation(depth)
        scene = build_scene(elevation, rgb, "DEMO_FALLBACK", "DEMO_FALLBACK", max_points=50)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "scene.json"
            export_scene_json(scene, out_path)
            self.assertTrue(out_path.exists())
            with open(out_path) as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["depth_status"], "DEMO_FALLBACK")


class TestBuildMeshScene(unittest.TestCase):
    """build_mesh_scene is the app's primary scene builder (a real surface,
    not a point cloud) -- see reconstruction.mesh's module docstring."""

    def _mesh_scene(self, seed=0, shape=(10, 10)):
        rng = np.random.default_rng(seed)
        elevation = rng.random(shape).astype(np.float32) * 3
        rgb = (rng.random(shape + (3,)) * 255).astype(np.uint8)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
        )
        mesh = build_terrain_mesh(grid, rgb, max_vertices=1000)
        return build_mesh_scene(grid, mesh)

    def test_scene_kind_is_terrain_mesh(self):
        scene = self._mesh_scene()
        self.assertEqual(scene["kind"], "terrain_mesh")

    def test_scene_is_tagged_uncalibrated(self):
        scene = self._mesh_scene()
        self.assertFalse(scene["calibrated"])
        self.assertFalse(scene["elevation_meta"]["calibrated"])
        self.assertEqual(scene["elevation_meta"]["unit"], "relative")
        self.assertEqual(scene["intrinsics_source"], "placeholder_uncalibrated")

    def test_scene_carries_depth_provenance(self):
        scene = self._mesh_scene()
        self.assertEqual(scene["depth_source"], "depth-anything-v2-small")
        self.assertEqual(scene["depth_status"], "SUCCESS")
        self.assertIn("elevation_convention", scene)

    def test_scene_elevation_meta_matches_grid_shape(self):
        scene = self._mesh_scene(shape=(12, 20))
        self.assertEqual(scene["elevation_meta"]["rows"], 12)
        self.assertEqual(scene["elevation_meta"]["cols"], 20)

    def test_scene_mesh_payload_is_present_and_json_serializable(self):
        scene = self._mesh_scene()
        self.assertIn("vertices", scene["mesh"])
        self.assertIn("triangles", scene["mesh"])
        self.assertIn("bbox_min", scene["mesh"])
        self.assertIn("bbox_max", scene["mesh"])
        # This is exactly what ships to the browser and the JSON download --
        # it must round-trip through json.dumps with no numpy leakage.
        dumped = json.dumps(scene)
        reloaded = json.loads(dumped)
        self.assertEqual(reloaded["kind"], "terrain_mesh")

    def test_export_mesh_scene_json_roundtrip(self):
        scene = self._mesh_scene(seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "mesh_scene.json"
            export_scene_json(scene, out_path)
            self.assertTrue(out_path.exists())
            with open(out_path) as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["kind"], "terrain_mesh")
            self.assertFalse(loaded["calibrated"])


if __name__ == "__main__":
    unittest.main()
