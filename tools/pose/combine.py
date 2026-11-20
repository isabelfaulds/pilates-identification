"""Combine extracted sequences without losing video boundaries or frame order."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def combine_csvs(input_dir, output):
    sources = sorted(input_dir.glob('*/frames.csv'))
    if not sources:
        raise ValueError(f'No per-video frames.csv files under {input_dir}')
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = None
    total = 0
    seen_ids = set()
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=output.parent,
                                         prefix='.combined-', delete=False) as destination:
            temp = Path(destination.name)
            writer = None
            for source in sources:
                summary = json.loads(source.with_name('summary.json').read_text())
                video_id = summary['video_id']
                if video_id in seen_ids:
                    raise ValueError(f'Duplicate video ID: {video_id}')
                seen_ids.add(video_id)
                video_name = Path(summary['source_file']).name
                with source.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    if schema is None:
                        schema = reader.fieldnames
                        required = {'video_id','exercise_name','frame_index','timestamp_s'}
                        if not schema or not required.issubset(schema) or 'video_name' in schema:
                            raise ValueError('Unexpected per-video CSV schema')
                        writer = csv.DictWriter(destination, fieldnames=['video_name']+schema)
                        writer.writeheader()
                    elif reader.fieldnames != schema:
                        raise ValueError(f'CSV schema mismatch: {source}')
                    last_frame, last_time = -1, -1.0
                    count = 0
                    for row in reader:
                        frame, timestamp = int(row['frame_index']), float(row['timestamp_s'])
                        if row['video_id'] != video_id or row['exercise_name'] != summary['exercise_name']:
                            raise ValueError(f'Video identity mismatch: {source}')
                        if frame <= last_frame or not math.isfinite(timestamp) or timestamp <= last_time:
                            raise ValueError(f'Frames are not chronological: {source}')
                        writer.writerow({'video_name':video_name, **row})
                        last_frame, last_time = frame, timestamp
                        count += 1
                    if not count or count != summary['processed_frames']:
                        raise ValueError(f'Frame count mismatch: {source}')
                    total += count
        os.replace(temp, output)
    finally:
        if temp and temp.exists():
            temp.unlink()
    return {'videos':len(seen_ids), 'frames':total, 'columns':len(schema)+1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/processed/landmarks')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/landmarks.csv')
    args = parser.parse_args()
    result = combine_csvs(args.input, args.output)
    print(f'Saved {args.output}: {result["frames"]:,} rows, {result["videos"]} videos, {result["columns"]} columns')


if __name__ == '__main__':
    main()
