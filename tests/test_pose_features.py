import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tools.pose.features import angle_degrees, row_features, derive_features, JOINTS, ANGLE_JOINTS


def example_row():
    row={'video_id':'sample','video_name':'sample.mp4','exercise_name':'Sample',
         'frame_index':'0','timestamp_s':'0','pose_detected':'1','extra_metadata':'preserved'}
    for joint in JOINTS:
        for axis,value in zip(('x','y','z'),(0.,0.,0.)):
            row[f'world_{joint}_{axis}']=str(value)
        for space in ('image','world'):
            for score in ('visibility','presence'):
                row[f'{space}_{joint}_{score}']='1'
    positions={'shoulder':(0,1,0),'elbow':(0,0,0),'wrist':(1,0,0),
               'hip':(0,-1,0),'knee':(1,-1,0),'ankle':(1,-2,0)}
    for side in ('left','right'):
        for joint,xyz in positions.items():
            for axis,value in zip(('x','y','z'),xyz):
                row[f'world_{side}_{joint}_{axis}']=str(value)
    return row


class AngleTests(unittest.TestCase):
    def test_known_angles(self):
        self.assertAlmostEqual(angle_degrees([1,0,0],[0,0,0],[0,1,0]),90)
        self.assertAlmostEqual(angle_degrees([-1,0,0],[0,0,0],[1,0,0]),180)
        self.assertAlmostEqual(angle_degrees([1,0,0],[0,0,0],[2,0,0]),0)

    def test_translation_and_uniform_scale_do_not_change_angle(self):
        points=np.array([[1,0,0],[0,0,0],[1,1,0]],dtype=float)
        self.assertAlmostEqual(angle_degrees(*points),45)
        self.assertAlmostEqual(angle_degrees(*(points*3+[10,-4,2])),45)

    def test_missing_or_degenerate_joint_is_not_zero_angle(self):
        self.assertTrue(math.isnan(angle_degrees([0,0,0],[0,0,0],[1,0,0])))
        self.assertTrue(math.isnan(angle_degrees([math.nan,0,0],[0,0,0],[1,0,0])))

    def test_all_six_triplets_and_confidence_mask(self):
        row=example_row()
        features=row_features(row)
        for name in ANGLE_JOINTS:
            self.assertAlmostEqual(features[name+'_deg'],90)
            self.assertEqual(features[name+'_valid'],1)
        row['image_left_wrist_visibility']='0.2'
        features=row_features(row)
        self.assertEqual(features['left_elbow_angle_deg'],'')
        self.assertEqual(features['left_elbow_angle_valid'],0)
        self.assertEqual(features['right_elbow_angle_valid'],1)
        row['world_right_wrist_presence']=''
        self.assertEqual(row_features(row)['right_elbow_angle_valid'],0)

    def test_no_detection_invalidates_all_angles(self):
        row=example_row();row['pose_detected']='0'
        features=row_features(row)
        self.assertTrue(all(features[name+'_valid']==0 for name in ANGLE_JOINTS))

    def test_csv_keeps_frames_metadata_and_blanks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.csv';output=root/'features.csv'
            first=example_row();second={**first,'frame_index':'1','timestamp_s':'0.033','pose_detected':'0'}
            with source.open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(first));writer.writeheader();writer.writerows([first,second])
            summary=derive_features(source,output)
            with output.open(newline='') as file:rows=list(csv.DictReader(file))
            self.assertEqual(summary['rows'],2)
            self.assertEqual(summary['videos'],1)
            for before,after in zip([first,second],rows):
                self.assertEqual(before,{key:after[key] for key in before})
            self.assertEqual(rows[1]['left_elbow_angle_deg'],'')
            self.assertEqual(rows[1]['left_elbow_angle_valid'],'0')
            self.assertTrue(output.with_suffix('.summary.json').is_file())


if __name__=='__main__':
    unittest.main()
