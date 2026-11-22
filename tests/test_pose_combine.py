import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.pose.combine import combine_csvs


class CombineTests(unittest.TestCase):
    def make_sequence(self, root, video_id, frames):
        folder = root/video_id
        folder.mkdir()
        (folder/'summary.json').write_text(json.dumps({
            'video_id':video_id, 'exercise_name':video_id,
            'source_file':f'/raw/{video_id}.mp4', 'processed_frames':len(frames)}))
        with (folder/'frames.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['video_id','exercise_name','frame_index','timestamp_s','image_nose_x'])
            for frame,timestamp,value in frames:
                writer.writerow([video_id,video_id,frame,timestamp,value])

    def test_combines_sequences_and_preserves_missing_values_and_time_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sequence(root,'a',[(0,0.,''),(1,.1,.5)])
            self.make_sequence(root,'b',[(0,0.,.2)])
            output = root/'combined.csv'
            result = combine_csvs(root, output)
            with output.open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(result, {'videos':2,'frames':3,'columns':6})
            self.assertEqual([r['video_name'] for r in rows], ['a.mp4','a.mp4','b.mp4'])
            self.assertEqual(rows[0]['image_nose_x'], '')
            self.assertEqual(rows[2]['timestamp_s'], '0.0')

    def test_invalid_sequence_does_not_replace_existing_combined_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sequence(root,'a',[(1,.1,.5),(0,0.,.2)])
            output = root/'combined.csv'
            output.write_text('previous valid output')
            with self.assertRaisesRegex(ValueError, 'not chronological'):
                combine_csvs(root, output)
            self.assertEqual(output.read_text(), 'previous valid output')


if __name__ == '__main__':
    unittest.main()
