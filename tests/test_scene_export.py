import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from depthwizard.reconstruction.dsm import depth_to_relative_elevation
from depthwizard.reconstruction.scene_export import build_scene, export_scene_json


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


if __name__ == "__main__":
    unittest.main()
