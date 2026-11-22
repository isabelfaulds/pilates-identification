"""Add confidence-masked joint angles to the combined landmark CSV."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .temporal import TemporalAngles
else:
    from temporal import TemporalAngles

ROOT = Path(__file__).resolve().parents[2]
ANGLE_JOINTS = {
    'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
    'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
    'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
    'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
    'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
    'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
}
JOINTS = sorted({joint for triplet in ANGLE_JOINTS.values() for joint in triplet})
FEATURE_COLUMNS = [column for name in ANGLE_JOINTS for column in (name+'_deg', name+'_valid')]


def angle_degrees(a, b, c):
    """Interior angle at b, from 0 to 180 degrees; undefined limbs return NaN."""
    a, b, c = (np.asarray(point, dtype=np.float64) for point in (a,b,c))
    if any(point.shape != (3,) for point in (a,b,c)):
        raise ValueError('Each joint must contain three world coordinates')
    u, v = a-b, c-b
    lengths = np.linalg.norm(u), np.linalg.norm(v)
    if not np.isfinite([a,b,c]).all() or min(lengths) <= 1e-8:
        return float('nan')
    cosine = np.dot(u/lengths[0], v/lengths[1])
    return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))


def number(value):
    return float(value) if value not in ('', None) else float('nan')


def row_features(row, confidence=.5):
    if row['pose_detected'] not in ('0','1'):
        raise ValueError('pose_detected must be 0 or 1')
    joints = {}
    for name in JOINTS:
        xyz = [number(row[f'world_{name}_{axis}']) for axis in ('x','y','z')]
        scores = [number(row[f'{space}_{name}_{score}'])
                  for space in ('image','world') for score in ('visibility','presence')]
        valid = (row['pose_detected'] == '1' and np.isfinite(xyz).all()
                 and all(math.isfinite(score) and score >= confidence for score in scores))
        joints[name] = xyz, valid
    result = {}
    for name,triplet in ANGLE_JOINTS.items():
        angle = angle_degrees(*(joints[joint][0] for joint in triplet)) if all(joints[joint][1] for joint in triplet) else float('nan')
        valid = math.isfinite(angle)
        result[name+'_deg'] = angle if valid else ''
        result[name+'_valid'] = int(valid)
    return result


def derive_features(source, output, confidence=.5, lag_seconds=(.1,.5,1.),
                    mean_seconds=(.5,1.), mean_lag_seconds=.5,
                    max_gap_seconds=.1, lag_tolerance_seconds=.05):
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Confidence must be between 0 and 1')
    if source.resolve() == output.resolve():
        raise ValueError('Write features to a separate file; preserve raw landmarks')
    temporal = TemporalAngles(ANGLE_JOINTS,lag_seconds,mean_seconds,mean_lag_seconds,
                              max_gap_seconds,lag_tolerance_seconds)
    added_columns = FEATURE_COLUMNS+temporal.columns
    temporal_counts = {column:0 for column in temporal.columns if column.endswith('_valid')}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    rows = 0
    valid_counts = dict.fromkeys(ANGLE_JOINTS, 0)
    video_counts = {}
    previous = {}
    try:
        with source.open(newline='') as infile, tempfile.NamedTemporaryFile(
                mode='w', newline='', dir=output.parent, prefix='.features-', delete=False) as outfile:
            temp = Path(outfile.name)
            reader = csv.DictReader(infile)
            required = {'video_id','video_name','frame_index','timestamp_s','pose_detected'}
            required.update(f'world_{joint}_{axis}' for joint in JOINTS for axis in ('x','y','z'))
            required.update(f'{space}_{joint}_{score}' for joint in JOINTS
                            for space in ('image','world') for score in ('visibility','presence'))
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError('Input must be a combined landmark CSV with image/world confidence scores')
            if set(added_columns).intersection(reader.fieldnames):
                raise ValueError('Input already contains angle feature columns')
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames+added_columns)
            writer.writeheader()
            for row in reader:
                video = row['video_id']
                frame, timestamp = int(row['frame_index']), float(row['timestamp_s'])
                last_frame,last_time = previous.get(video,(-1,-1.))
                if frame <= last_frame or not math.isfinite(timestamp) or timestamp <= last_time:
                    raise ValueError(f'Non-chronological frame in {video}')
                features = row_features(row, confidence)
                features.update(temporal.update(video,timestamp,features,row.get('segment_id')))
                writer.writerow({**row, **features})
                for column in temporal_counts:
                    temporal_counts[column] += features[column]
                previous[video] = frame,timestamp
                rows += 1
                video_counts[video] = video_counts.get(video,0)+1
                for name in valid_counts:
                    valid_counts[name] += features[name+'_valid']
            if not rows:
                raise ValueError('Input contains no frames')
        os.replace(temp,output)
    finally:
        if temp and temp.exists():
            temp.unlink()
    with source.open('rb') as infile:
        source_hash = hashlib.file_digest(infile,'sha256').hexdigest()
    summary = {
        'schema_version':3, 'source_file':str(source.resolve()),
        'source_sha256':source_hash, 'rows':rows, 'videos':len(video_counts),
        'video_frame_counts':video_counts, 'coordinate_space':'MediaPipe estimated 3D world coordinates',
        'angle_units':'degrees', 'angle_range':[0,180], 'joint_order':'a-b-c; angle measured at b',
        'joint_triplets':ANGLE_JOINTS, 'confidence_threshold':confidence,
        'validity_rule':'all three joints require finite world XYZ and image/world visibility/presence >= threshold; pose must be detected',
        'valid_angle_counts':valid_counts,
        'valid_angle_fractions':{name:count/rows for name,count in valid_counts.items()},
        'added_columns':added_columns,
        'temporal':{'lag_seconds':list(lag_seconds),'trailing_mean_seconds':list(mean_seconds),
                    'mean_lag_seconds':mean_lag_seconds,'max_gap_seconds':max_gap_seconds,
                    'lag_tolerance_seconds':lag_tolerance_seconds,
                    'lag_selection':'latest observation at or before t-lag, within tolerance; no interpolation',
                    'mean':'arithmetic mean of samples in inclusive [t-window,t], requiring a full continuous valid history',
                    'angle_velocity':'(current angle - previous angle) / actual elapsed seconds; backward difference only',
                    'angle_velocity_units':'degrees per second; positive = increasing interior angle, negative = decreasing',
                    'reset':'per video/segment; any invalid angle resets that angle; long frame gaps reset all angles',
                    'valid_counts':temporal_counts},
        'missing_angles':'blank with validity flag 0; no interpolation or row removal',
    }
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/processed/landmarks.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/features/frames.csv')
    parser.add_argument('--confidence',type=float,default=.5)
    parser.add_argument('--lag-seconds',type=float,nargs='+',default=[.1,.5,1.])
    parser.add_argument('--mean-seconds',type=float,nargs='+',default=[.5,1.])
    parser.add_argument('--mean-lag-seconds',type=float,default=.5)
    parser.add_argument('--max-gap-seconds',type=float,default=.1)
    parser.add_argument('--lag-tolerance-seconds',type=float,default=.05)
    args = parser.parse_args()
    result = derive_features(args.input,args.output,args.confidence,args.lag_seconds,
                             args.mean_seconds,args.mean_lag_seconds,args.max_gap_seconds,
                             args.lag_tolerance_seconds)
    print(f'Saved {args.output}: {result["rows"]:,} frames from {result["videos"]} videos; {len(result["added_columns"])} angle/temporal columns including validity flags')


if __name__ == '__main__':
    main()
