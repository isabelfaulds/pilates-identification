"""Small end-to-end check; skipped if Google's optional model isn't downloaded."""
import csv
from fractions import Fraction
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np

from tools.pose.extract import ROOT, extract, metadata_for


@unittest.skipUnless((ROOT/'models/pose_landmarker_full.task').is_file(), 'Download pose model to run integration check')
class PoseIntegrationTest(unittest.TestCase):
    def test_variable_rate_video_missing_poses_and_safe_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root/'uncataloged.mp4'
            with av.open(str(video), 'w') as output:
                stream = output.add_stream('libx264', rate=25)
                stream.width = stream.height = 256
                stream.pix_fmt = 'yuv420p'
                stream.time_base = Fraction(1,1000)
                for timestamp in [0,40,80,200,240,400,440]:
                    frame = av.VideoFrame.from_ndarray(np.zeros((256,256,3),dtype=np.uint8),format='rgb24')
                    frame.pts = timestamp
                    frame.time_base = Fraction(1,1000)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            args = SimpleNamespace(output=root/'processed', model=ROOT/'models/pose_landmarker_full.task',
                                   sample_fps=10., confidence=.5, overwrite=False)
            info = metadata_for(video, [])
            summary = extract(video, info, args)
            out = args.output/info['video_id']
            with np.load(out/'sequence.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['frame_index'], [0,3,5])
                np.testing.assert_allclose(data['timestamp_s'], [0.,.2,.4], atol=1e-8)
                self.assertEqual(data['image_landmarks'].shape, (3,33,5))
                self.assertTrue(np.isnan(data['image_landmarks']).all())
                self.assertTrue(np.isnan(data['world_landmarks']).all())
                self.assertFalse(data['pose_detected'].any())
                self.assertFalse(data['landmark_valid'].any())
            with (out/'frames.csv').open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]), 338)
            self.assertTrue(all(row['image_left_knee_x'] == '' for row in rows))
            self.assertEqual(summary['decoded_frames'], 7)
            self.assertEqual(summary['label_status'], 'filename_only_not_frame_verified')
            self.assertTrue((out/'preview.jpg').is_file())
            with patch('tools.pose.extract.vision.PoseLandmarker.create_from_options', side_effect=AssertionError('Cache should skip inference')):
                self.assertEqual(extract(video, info, args), summary)
            args.confidence = .8
            with self.assertRaisesRegex(FileExistsError, 'different inputs/settings'):
                extract(video, info, args)


if __name__ == '__main__':
    unittest.main()
