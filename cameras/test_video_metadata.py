"""Unit tests for the video metadata sidecar (stdlib only; no camera hardware).

Run with: python3 -m pytest test_video_metadata.py -v
"""
import json
import os
import tempfile
import unittest

import video_metadata as vm


class TestVideoMetadata(unittest.TestCase):
    def test_sidecar_path(self):
        self.assertEqual(vm.sidecar_path("/data/clip_zCam.mp4"),
                         "/data/clip_zCam.json")

    def test_measured_fps(self):
        # 6810 frames over 60 s -> 113.5 fps (the true rate the container lacks).
        meta = vm.build_metadata("clip.mp4", 6810, 1000.0, 1060.0,
                                 camera="zCam", width=400, height=116,
                                 nominal_fps=30.0)
        self.assertAlmostEqual(meta["measured_fps"], 113.5, places=3)
        self.assertAlmostEqual(meta["duration_s"], 60.0, places=6)
        self.assertEqual(meta["nominal_fps"], 30.0)
        self.assertEqual(meta["schema"], "mastqg.video_sidecar.v1")

    def test_zero_and_bad_duration_guarded(self):
        self.assertIsNone(vm.build_metadata("c.mp4", 100, 5.0, 5.0)["measured_fps"])
        self.assertIsNone(vm.build_metadata("c.mp4", 100, None, 1.0)["measured_fps"])

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            vp = os.path.join(d, "diamond1_zCam.mp4")
            open(vp, "wb").close()
            p = vm.write_sidecar(vp, 1000, 0.0, 10.0, camera="zCam")
            self.assertTrue(p and os.path.exists(p))
            meta = json.load(open(p))
            self.assertAlmostEqual(meta["measured_fps"], 100.0)

    def test_write_never_raises(self):
        # A bad path must be swallowed (returns None), never interrupt recording.
        self.assertIsNone(
            vm.write_sidecar("/no/such/dir/x.mp4", 10, 0.0, 1.0))
        self.assertIsNone(vm.write_sidecar(None, 10, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
