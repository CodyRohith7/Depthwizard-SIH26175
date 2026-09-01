import json
import unittest
import warnings

import numpy as np

from depthwizard.reconstruction.elevation_grid import build_elevation_grid
from depthwizard.reconstruction.mesh import (
    build_terrain_mesh,
    compute_elevation_colors,
    mesh_to_json_safe,
)


def _grid(elevation, **kwargs):
    return build_elevation_grid(
        elevation.astype(np.float32), depth_source="depth-anything-v2-small",
        depth_status="SUCCESS", **kwargs,
    )


class TestVertexGeneration(unittest.TestCase):
    def test_full_resolution_when_under_budget(self):
        rng = np.random.default_rng(0)
        elevation = rng.random((6, 8)).astype(np.float32) * 10
        rgb = (rng.random((6, 8, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertEqual(mesh["stride"], 1)
        self.assertEqual((mesh["grid_rows"], mesh["grid_cols"]), (6, 8))
        self.assertEqual(mesh["vertex_count"], 6 * 8)
        self.assertEqual(mesh["vertices"].shape, (48, 3))

    def test_decimation_reduces_vertex_count_for_large_grid(self):
        rng = np.random.default_rng(1)
        elevation = rng.random((100, 100)).astype(np.float32) * 5
        rgb = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=500)
        self.assertGreater(mesh["stride"], 1)
        # Decimation is stride-based, so the result can slightly overshoot
        # the target -- but must stay well below the undecimated 10000.
        self.assertLessEqual(mesh["vertex_count"], 1000)

    def test_raises_for_too_small_image_after_decimation(self):
        elevation = np.ones((1, 5), dtype=np.float32)
        rgb = np.zeros((1, 5, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)

    def test_raises_when_no_finite_values(self):
        elevation = np.full((4, 4), np.nan, dtype=np.float32)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)


class TestTriangleConnectivity(unittest.TestCase):
    def test_fully_finite_grid_forms_two_triangles_per_cell(self):
        rng = np.random.default_rng(2)
        elevation = rng.random((6, 8)).astype(np.float32)
        rgb = (rng.random((6, 8, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        expected = 2 * (6 - 1) * (8 - 1)
        self.assertEqual(mesh["triangle_count"] + mesh["degenerate_triangles_dropped"], expected)

    def test_triangle_indices_are_in_range(self):
        rng = np.random.default_rng(3)
        elevation = rng.random((10, 10)).astype(np.float32)
        rgb = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertTrue(np.all(mesh["triangles"] >= 0))
        self.assertTrue(np.all(mesh["triangles"] < mesh["vertex_count"]))


class TestHoleHandling(unittest.TestCase):
    def test_no_triangle_references_a_nodata_vertex(self):
        rng = np.random.default_rng(4)
        elevation = rng.random((10, 10)).astype(np.float32) * 3 + 100.0
        elevation[3:6, 3:6] = np.nan
        rgb = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
        grid = _grid(elevation)
        mesh = build_terrain_mesh(grid, rgb, max_vertices=1000)
        sc = mesh["grid_cols"]
        nodata_idx = {
            r * sc + c for r in range(3, 6) for c in range(3, 6)
        }
        referenced = set(mesh["triangles"].reshape(-1).tolist())
        self.assertEqual(nodata_idx & referenced, set())

    def test_holes_do_not_crash_normal_or_color_computation(self):
        rng = np.random.default_rng(5)
        elevation = rng.random((10, 10)).astype(np.float32) * 3 + 100.0
        elevation[0:3, 0:3] = np.nan
        rgb = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertTrue(np.all(np.isfinite(mesh["normals"])))
        self.assertTrue(np.all(np.isfinite(mesh["vertices"])))


class TestBoundingBox(unittest.TestCase):
    def test_bbox_ignores_unreferenced_nodata_placeholder_vertices(self):
        # Regression test: nodata vertices get an artificial (0,0,0)
        # position so the vertex buffer stays finite. A bbox computed over
        # ALL vertices (rather than only triangle-referenced ones) would be
        # pulled toward the origin even though every real elevation value
        # here is around 100 -- which would corrupt camera fit/centering.
        rng = np.random.default_rng(6)
        elevation = np.full((10, 10), 100.0, dtype=np.float32)
        elevation += rng.random((10, 10)).astype(np.float32)
        elevation[0:3, 0:3] = np.nan
        rgb = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertGreater(mesh["bbox_min"][2], 50.0)
        self.assertLess(mesh["bbox_max"][2], 200.0)


class TestNormals(unittest.TestCase):
    def test_referenced_vertex_normals_are_unit_length(self):
        rng = np.random.default_rng(7)
        elevation = rng.random((10, 10)).astype(np.float32) * 4
        rgb = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        referenced = np.unique(mesh["triangles"].reshape(-1))
        lengths = np.linalg.norm(mesh["normals"][referenced], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


class TestDegenerateTriangleRejection(unittest.TestCase):
    def test_a_single_wild_spike_is_dropped(self):
        rng = np.random.default_rng(8)
        elevation = rng.random((10, 10)).astype(np.float32)
        elevation[5, 5] = 5000.0
        rgb = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertGreater(mesh["degenerate_triangles_dropped"], 0)
        # The rest of the mesh should still be usable.
        self.assertGreater(mesh["triangle_count"], 0)


class TestColorAndTextureCoordinates(unittest.TestCase):
    def test_colors_rgb_shape_and_dtype(self):
        rng = np.random.default_rng(9)
        elevation = rng.random((8, 8)).astype(np.float32)
        rgb = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertEqual(mesh["colors_rgb"].dtype, np.uint8)
        self.assertEqual(mesh["colors_rgb"].shape, (mesh["vertex_count"], 3))

    def test_colors_elevation_shape_and_dtype(self):
        rng = np.random.default_rng(10)
        elevation = rng.random((8, 8)).astype(np.float32)
        rgb = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        self.assertEqual(mesh["colors_elevation"].dtype, np.uint8)
        self.assertEqual(mesh["colors_elevation"].shape, (mesh["vertex_count"], 3))

    def test_elevation_colors_are_nan_safe(self):
        # Regression test: plain np.min/np.max propagate NaN across the
        # *entire* output the instant any single NaN is present, silently
        # corrupting every vertex's color, not just the nodata ones.
        elevation = np.array([0.0, 1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            colors = compute_elevation_colors(elevation)
        self.assertTrue(np.all(np.isfinite(colors)))
        self.assertEqual(colors.shape, (5, 3))

    def test_all_nan_elevation_colors_returns_zeros(self):
        elevation = np.full(9, np.nan, dtype=np.float32)
        colors = compute_elevation_colors(elevation)
        self.assertTrue(np.all(colors == 0))


class TestSceneMetadata(unittest.TestCase):
    def test_mesh_carries_honest_provenance_fields(self):
        rng = np.random.default_rng(11)
        elevation = rng.random((8, 8)).astype(np.float32)
        rgb = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000, assumed_hfov_deg=75.0)
        self.assertFalse(mesh["calibrated"])
        self.assertEqual(mesh["intrinsics_source"], "placeholder_uncalibrated")
        self.assertEqual(mesh["assumed_hfov_deg"], 75.0)
        self.assertEqual(mesh["source_rows"], 8)
        self.assertEqual(mesh["source_cols"], 8)

    def test_mesh_to_json_safe_produces_plain_python_types(self):
        rng = np.random.default_rng(12)
        elevation = rng.random((8, 8)).astype(np.float32)
        rgb = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        mesh = build_terrain_mesh(_grid(elevation), rgb, max_vertices=1000)
        safe = mesh_to_json_safe(mesh)
        self.assertIsInstance(safe["vertices"], list)
        self.assertIsInstance(safe["triangles"], list)
        self.assertIsInstance(safe["bbox_min"], list)
        self.assertIsInstance(safe["bbox_max"], list)
        # Must be directly JSON-serializable -- this is what actually ships
        # to the browser and to the "Download scene JSON" export.
        json.dumps(safe)


if __name__ == "__main__":
    unittest.main()
