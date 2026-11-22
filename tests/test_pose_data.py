"""Data semantics that must remain stable for downstream temporal models."""
import unittest
import tempfile
from types import SimpleNamespace

import numpy as np
from tools.pose.extract import arrays_from_result, quality_mask, missing_intervals, metadata_for, discover_videos
from pathlib import Path


class PoseDataTests(unittest.TestCase):
    def test_missing_pose_remains_nan(self):
        image, world, detected = arrays_from_result(SimpleNamespace(pose_landmarks=[]))
        self.assertFalse(detected)
        self.assertEqual(image.shape, (33, 5))
        self.assertTrue(np.isnan(image).all())
        self.assertTrue(np.isnan(world).all())
        self.assertFalse(quality_mask(image, .5).any())

    def test_low_confidence_values_are_preserved_but_masked(self):
        landmarks = [SimpleNamespace(x=.2,y=.3,z=-.1,visibility=.9,presence=.8) for _ in range(33)]
        landmarks[25] = SimpleNamespace(x=.7,y=.6,z=.2,visibility=.2,presence=.9)
        result = SimpleNamespace(pose_landmarks=[landmarks], pose_world_landmarks=[landmarks])
        image, world, detected = arrays_from_result(result)
        self.assertTrue(detected)
        mask = quality_mask(image, .5)
        self.assertEqual(mask.sum(), 32)
        self.assertFalse(mask[25])
        self.assertAlmostEqual(float(image[25,0]), .7)

    def test_presence_is_required_as_well_as_visibility(self):
        points = np.ones((2,33,5), dtype=np.float32)
        points[1,4,4] = .1
        self.assertFalse(quality_mask(points, .5)[1,4])

    def test_missing_intervals_preserve_time_gaps(self):
        times = np.array([0.,.1,.3,.4,.5])
        detected = np.array([False,False,True,True,False])
        intervals = missing_intervals(times, detected)
        np.testing.assert_allclose(intervals, [[0.,.3],[.5,.6]])

    def test_ambiguous_names_cannot_silently_change_identity(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            metadata_for(Path('bear.mp4'), [{'name':'Bear'}, {'name':'Bear'}])

    def test_recursive_discovery_includes_nested_and_uppercase_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'session').mkdir()
            for name in ['a.mp4','session/b.MOV','session/c.webm','notes.txt','.gitkeep']:
                (root/name).touch()
            self.assertEqual([p.name for p in discover_videos([],root)], ['a.mp4','b.MOV','c.webm'])

    def test_generic_video_without_catalog_has_stable_unique_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'one').mkdir(); (root/'two').mkdir()
            a = root/'one'/'exercise.mp4'; a.write_bytes(b'first recording')
            b = root/'two'/'exercise.mp4'; b.write_bytes(b'second recording')
            first = metadata_for(a, [])
            self.assertEqual(first, metadata_for(a, []))
            self.assertNotEqual(first['video_id'], metadata_for(b, [])['video_id'])
            self.assertEqual(first['source_url'], '')

    def test_explicit_missing_video_does_not_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, 'Not a supported video'):
            discover_videos([Path('/nonexistent/missing.mp4')], Path('/unused'))


if __name__ == '__main__':
    unittest.main()
