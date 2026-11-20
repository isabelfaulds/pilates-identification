"""Extract chronological, unmodified MediaPipe landmarks from local videos."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
VIDEO_SUFFIXES = {'.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'}
NAMES = (
    'nose left_eye_inner left_eye left_eye_outer right_eye_inner right_eye '
    'right_eye_outer left_ear right_ear mouth_left mouth_right left_shoulder '
    'right_shoulder left_elbow right_elbow left_wrist right_wrist left_pinky '
    'right_pinky left_index right_index left_thumb right_thumb left_hip right_hip '
    'left_knee right_knee left_ankle right_ankle left_heel right_heel '
    'left_foot_index right_foot_index'
).split()
CHANNELS = ['x', 'y', 'z', 'visibility', 'presence']
CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
               (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),
               (27,31),(28,30),(30,32),(28,32)]


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def filename(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_',
                  re.sub(r'\s+', '_', name.strip().lower())).rstrip('. ') + '.mp4'


def metadata_for(video, catalog):
    matches = [row for row in catalog if filename(row['name']) == video.name]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous catalog name: {video.name}')
    if matches:
        row = matches[0]
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
            raise ValueError('Catalog video ID must be a safe directory name')
        return {'video_id': row['id'], 'exercise_name': row['name'],
                'source_url': row['url'], 'candidate_groups': row.get('group', '')}
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', video.stem) or 'video'
    return {'video_id': f'{stem}-{sha256(video)[:12]}', 'exercise_name': video.stem,
            'source_url': '', 'candidate_groups': ''}


def discover_videos(inputs, input_dir):
    """Recursively discover videos; ignore unrelated files, not directory names."""
    if not inputs and not input_dir.is_dir():
        raise ValueError(f'Input folder does not exist: {input_dir}')
    paths = inputs or input_dir.rglob('*')
    videos = set()
    for candidate in paths:
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            videos.add(candidate.resolve())
        elif inputs:
            raise ValueError(f'Not a supported video file: {candidate}')
    return sorted(videos)


def arrays_from_result(result):
    """No pose is NaN, never a zero-valued skeleton or a dropped frame."""
    image = np.full((33, 5), np.nan, dtype=np.float32)
    world = image.copy()
    detected = bool(result.pose_landmarks)
    if detected:
        if len(result.pose_landmarks[0]) != 33 or not result.pose_world_landmarks:
            raise ValueError('Unexpected MediaPipe landmark schema')
        for destination, landmarks in [(image, result.pose_landmarks[0]),
                                       (world, result.pose_world_landmarks[0])]:
            if len(landmarks) != 33:
                raise ValueError('Expected 33 landmarks')
            for index, landmark in enumerate(landmarks):
                destination[index] = [getattr(landmark, key, np.nan) for key in CHANNELS]
    return image, world, detected


def quality_mask(image, confidence):
    return (np.isfinite(image).all(axis=-1) &
            (image[..., 3] >= confidence) & (image[..., 4] >= confidence))


def missing_intervals(times, detected):
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    gaps = []
    start = None
    for index, found in enumerate(detected):
        if not found and start is None:
            start = float(times[index])
        if found and start is not None:
            gaps.append([start, float(times[index])])
            start = None
    if start is not None:
        gaps.append([start, float(times[-1]) + step])
    return gaps


def render_preview(video, output, indices, times, landmarks, valid):
    """Six frames across the sequence, with only confident body joints drawn."""
    picks = np.unique(np.linspace(0, len(indices) - 1, 6, dtype=int))
    wanted = {int(indices[i]): int(i) for i in picks}
    panels = {}
    with av.open(str(video)) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if index not in wanted:
                continue
            i = wanted[index]
            image = frame.to_image().convert('RGB')
            draw = ImageDraw.Draw(image)
            points = [(float(x)*image.width, float(y)*image.height) for x,y in landmarks[i,:,:2]]
            for a,b in CONNECTIONS:
                if valid[i,a] and valid[i,b]:
                    draw.line([points[a], points[b]], fill='#00ff80', width=4)
            for j in range(11,33):
                if valid[i,j]:
                    x,y = points[j]
                    draw.ellipse((x-4,y-4,x+4,y+4), fill='#ffcc00')
            panel = Image.new('RGB', (400,432), '#161616')
            panel.paste(ImageOps.pad(image, (400,400)), (0,0))
            ImageDraw.Draw(panel).text((8,410), f'{times[i]:.2f}s | frame {index} | {valid[i].sum()}/33 confident joints', fill='white')
            panels[i] = panel
            if len(panels) == len(wanted):
                break
    canvas = Image.new('RGB', (1200,864), '#161616')
    for slot,i in enumerate(picks):
        canvas.paste(panels[int(i)], ((slot % 3)*400, (slot // 3)*432))
    canvas.save(output, quality=90)


def extract(video, info, args):
    out = args.output / info['video_id']
    if out.exists() and not args.overwrite:
        try:
            prior = json.loads((out / 'summary.json').read_text())
            complete = all((out/name).is_file() and (out/name).stat().st_size > 0
                           for name in ['frames.csv','sequence.npz','preview.jpg'])
            current = (complete and prior.get('schema_version') == 1 and
                       prior.get('source_sha256') == sha256(video) and
                       prior.get('source_file') == str(video.resolve()) and
                       prior.get('extractor_sha256') == sha256(Path(__file__)) and
                       prior.get('model_sha256') == sha256(args.model) and
                       prior.get('sample_fps') == args.sample_fps and
                       prior.get('confidence_threshold') == args.confidence and
                       all(prior.get(key) == value for key,value in info.items()) and
                       prior.get('versions') == {name: importlib.metadata.version(name)
                                                for name in ['mediapipe','av','numpy','pillow']})
        except (OSError, ValueError):
            current = False
        if current:
            print(f'Already processed: {video.name}', flush=True)
            return prior
        raise FileExistsError(f'{out} is incomplete or from different inputs/settings; use --overwrite')
    args.output.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.extract-', dir=args.output))
    try:
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model),
                                            delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        indices, pts, times, api_times, found, images, worlds = [],[],[],[],[],[],[]
        last_timestamp = -1
        last_bucket = -1
        first_time = previous_time = None
        decoded_count = 0
        fields = ['video_id','exercise_name','frame_index','source_pts','timestamp_s',
                  'mediapipe_timestamp_ms','pose_detected','confident_landmarks']
        fields += [f'{space}_{name}_{channel}' for space in ['image','world']
                   for name in NAMES for channel in CHANNELS]
        with av.open(str(video)) as source, vision.PoseLandmarker.create_from_options(options) as model, \
                (temp / 'frames.csv').open('w', newline='') as csvfile:
            stream = source.streams.video[0]
            stream_info = {'width': stream.width, 'height': stream.height,
                           'reported_fps': float(stream.average_rate) if stream.average_rate else None,
                           'time_base': str(stream.time_base),
                           'reported_frames': stream.frames}
            writer = csv.writer(csvfile)
            writer.writerow(fields)
            for index, frame in enumerate(source.decode(video=0)):
                decoded_count += 1
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f'Frame {index} has no presentation timestamp')
                raw_time = float(frame.pts * frame.time_base)
                if first_time is None:
                    first_time = raw_time
                t = raw_time - first_time
                if previous_time is not None and t <= previous_time:
                    raise ValueError('Non-increasing source presentation timestamps')
                previous_time = t
                if args.sample_fps:
                    bucket = math.floor(t * args.sample_fps + 1e-8)
                    if bucket <= last_bucket:
                        continue
                    last_bucket = bucket
                # MediaPipe requires strictly increasing integer milliseconds.
                # Keep actual PTS-derived seconds separately; do not alter source time.
                api_timestamp = max(last_timestamp + 1, round(t * 1000))
                last_timestamp = api_timestamp
                rgb = frame.to_ndarray(format='rgb24')
                result = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                         data=np.ascontiguousarray(rgb)), api_timestamp)
                image, world, detected = arrays_from_result(result)
                valid = quality_mask(image, args.confidence)
                writer.writerow([info['video_id'],info['exercise_name'],index,frame.pts,
                                 f'{t:.9f}',api_timestamp,int(detected),int(valid.sum())] +
                                [float(v) if np.isfinite(v) else '' for v in
                                 np.concatenate([image.ravel(), world.ravel()])])
                indices.append(index); pts.append(frame.pts); times.append(t)
                api_times.append(api_timestamp); found.append(detected)
                images.append(image); worlds.append(world)
                if len(times) % 300 == 0:
                    print(f'  {video.name}: {len(times)} frames processed, {t:.1f}s', flush=True)
        if not times:
            raise ValueError('Video contained no decodable frames')
        image_array = np.stack(images)
        world_array = np.stack(worlds)
        valid = quality_mask(image_array, args.confidence)
        timestamp_array = np.array(times, dtype=np.float64)
        detected_array = np.array(found, dtype=bool)
        np.savez_compressed(temp / 'sequence.npz',
                            video_id=np.array(info['video_id']),
                            exercise_name=np.array(info['exercise_name']),
                            frame_index=np.array(indices, dtype=np.int64),
                            source_pts=np.array(pts, dtype=np.int64),
                            timestamp_s=timestamp_array,
                            mediapipe_timestamp_ms=np.array(api_times, dtype=np.int64),
                            pose_detected=detected_array,
                            landmark_valid=valid,
                            image_landmarks=image_array, world_landmarks=world_array,
                            landmark_names=np.array(NAMES), channels=np.array(CHANNELS))
        summary = {
            'schema_version': 1, **info, 'source_file': str(video.resolve()),
            'source_sha256': sha256(video), 'model_sha256': sha256(args.model),
            'extractor_sha256': sha256(Path(__file__)),
            'model_file': args.model.name,
            'versions': {name: importlib.metadata.version(name) for name in ['mediapipe','av','numpy','pillow']},
            'video': stream_info, 'timestamp_origin_source_s': first_time,
            'decoded_frames': decoded_count, 'processed_frames': len(times),
            'sample_fps': args.sample_fps, 'last_timestamp_s': times[-1],
            'detected_frames': int(detected_array.sum()),
            'detection_rate': float(detected_array.mean()),
            'confidence_threshold': args.confidence,
            'confident_landmark_fraction': float(valid.mean()),
            'body_landmark_confident_fraction': float(valid[:,11:].mean()),
            'missing_pose_intervals_s': missing_intervals(timestamp_array, detected_array),
            'per_landmark_confident_fraction': dict(zip(NAMES, valid.mean(axis=0).tolist())),
            'label_status': ('video_title_only_not_frame_verified' if info['source_url']
                             else 'filename_only_not_frame_verified'),
            'single_person_assumption': True,
            'processing': {'running_mode':'VIDEO','delegate':'CPU','num_poses':1,
                           'detection_confidence':0.5,'presence_confidence':0.5,'tracking_confidence':0.5,
                           'interpolation':False,'normalization':False,'smoothing':'MediaPipe VIDEO internal tracking only'},
        }
        (temp / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        render_preview(video, temp / 'preview.jpg', indices, times, image_array, valid)
        if out.exists():
            shutil.rmtree(out)
        temp.rename(out)
        print(f"Saved {out}: {len(times)} frames, pose detected {summary['detection_rate']:.1%}", flush=True)
        return summary
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='*', type=Path, help='Optional explicit video paths instead of scanning the input folder')
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/raw', help='Recursively scan this folder (default: data/raw)')
    parser.add_argument('--model', type=Path, default=ROOT/'models/pose_landmarker_full.task')
    parser.add_argument('--catalog', type=Path, default=ROOT/'output/lagree/micro-exercises.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/landmarks')
    parser.add_argument('--sample-fps', type=float, help='Optional maximum sampling rate; default keeps every frame')
    parser.add_argument('--confidence', type=float, default=0.5, help='Visibility AND presence threshold for validity mask')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.sample_fps is not None and (not math.isfinite(args.sample_fps) or args.sample_fps <= 0):
        parser.error('--sample-fps must be finite and positive')
    if not 0 <= args.confidence <= 1:
        parser.error('--confidence must be between 0 and 1')
    if not args.model.is_file():
        parser.error('Pose model missing. See docs/pose-pipeline.md for the model download command.')
    try:
        videos = discover_videos(args.videos, args.input_dir)
    except ValueError as error:
        parser.error(str(error))
    if not videos:
        parser.error('No local videos found. Download selected videos first.')
    catalog = []
    if args.catalog.is_file():
        with args.catalog.open(newline='') as source:
            catalog = list(csv.DictReader(source))
    jobs = [(video, metadata_for(video, catalog)) for video in videos]
    ids = [info['video_id'] for _,info in jobs]
    if len(set(ids)) != len(ids):
        parser.error('Multiple inputs resolve to the same video ID')
    failures = []
    for video,info in jobs:
        try:
            extract(video, info, args)
        except Exception as error:
            print(f'Failed {video.name}: {error}', flush=True)
            failures.append(video.name)
    if failures:
        raise SystemExit(f'{len(failures)} video(s) failed: {", ".join(failures)}')


if __name__ == '__main__':
    main()
