import unittest

import numpy as np

from tools.pose.train_lstm import (sampled_indices,split_candidates,assert_protected_splits,
                                   fit_scaler,transform,Video,window_features,VALUE_COLUMNS)
from tools.pose.features import row_features
from tests.test_pose_features import example_row


class TrainingDataTests(unittest.TestCase):
    def test_protected_blocks_have_no_overlap_or_shared_frames(self):
        parts=split_candidates(np.arange(0,30,.5),42)
        manifest=[]
        for split,starts in parts.items():
            for start in starts:
                manifest.append({'split':split,'video_id':'a','start_s':start,'end_s':start+2,
                                 'source_frame_indices':list(range(round(start*30),round(start*30)+60))})
        assert_protected_splits(manifest)
        self.assertEqual(len(parts['test']),2)
        self.assertEqual(len(parts['validation']),1)
        self.assertGreater(len(parts['train']),0)
        self.assertEqual(parts,split_candidates(np.arange(0,30,.5),42))

    def test_leaking_split_is_rejected(self):
        a={'split':'train','video_id':'a','start_s':0,'end_s':2,'source_frame_indices':[0,1]}
        b={'split':'test','video_id':'a','start_s':1,'end_s':3,'source_frame_indices':[1,2]}
        with self.assertRaisesRegex(ValueError,'overlap'):assert_protected_splits([a,b])

    def test_resampling_does_not_use_future_frames_or_cross_large_gaps(self):
        times=np.arange(100)/30
        indices=sampled_indices(times,.1)
        np.testing.assert_array_equal(indices,np.arange(3,63))
        self.assertIsNone(sampled_indices(np.array([0.,.5,1.,1.5,2.]),0))

    def test_training_only_scaler_and_missing_indicators(self):
        train=np.array([[[1.,np.nan],[3.,np.nan]]],dtype=np.float32)
        mean,scale,counts=fit_scaler(train)
        np.testing.assert_allclose(mean,[2,0]);np.testing.assert_allclose(scale,[1,1])
        np.testing.assert_array_equal(counts,[2,0])
        test=transform(np.array([[[100.,np.nan]]]),mean,scale)
        np.testing.assert_allclose(test,[[[98,0,1,0]]])
        np.testing.assert_allclose(mean,[2,0])

    def test_each_snippet_resets_feature_history_and_noise_is_reproducible(self):
        times=np.arange(120)/30
        rows=[]
        for i,t in enumerate(times):
            row=example_row();row['timestamp_s']=str(t);row['frame_index']=str(i);rows.append(row)
        video=Video('a','a.mp4','Bear',rows,times,[row_features(row) for row in rows])
        indices=sampled_indices(times,1.)
        clean=window_features(video,indices)
        self.assertEqual(clean.shape,(60,54))
        self.assertEqual(np.isfinite(clean[0]).sum(),6)
        self.assertTrue(np.isfinite(clean[-1]).all())
        noise1=window_features(video,indices,.005,42)
        noise2=window_features(video,indices,.005,42)
        np.testing.assert_allclose(noise1,noise2,equal_nan=True)
        self.assertFalse(np.allclose(clean,noise1,equal_nan=True))
        self.assertEqual(len(VALUE_COLUMNS),54)


if __name__=='__main__':unittest.main()
