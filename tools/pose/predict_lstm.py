"""Classify one two-second snippet from an extracted landmark CSV."""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .train_lstm import ROOT,VALUE_COLUMNS,load_videos,sampled_indices,window_features,transform
else:
    from train_lstm import ROOT,VALUE_COLUMNS,load_videos,sampled_indices,window_features,transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Combined raw landmark CSV, not an RGB video')
    parser.add_argument('--experiment',type=Path,required=True,help='Folder containing model.keras and preprocessing.json')
    parser.add_argument('--video-id',help='Required if the input contains multiple videos')
    parser.add_argument('--start',type=float,required=True,help='Start time in source-video seconds')
    args=parser.parse_args()
    configuration=json.loads((args.experiment/'preprocessing.json').read_text())
    if configuration['value_columns']!=VALUE_COLUMNS:
        parser.error('Saved model feature schema differs from this feature code')
    videos=load_videos(args.input,None)
    if args.video_id:
        if args.video_id not in videos:parser.error('Video ID not found in input')
        video=videos[args.video_id]
    elif len(videos)==1:
        video=next(iter(videos.values()))
    else:parser.error('Specify --video-id when input contains multiple videos')
    indices=sampled_indices(video.timestamps,args.start)
    if indices is None:parser.error('Not enough regularly spaced source frames for this snippet')
    raw=window_features(video,indices)[None,...]
    features=transform(raw,np.array(configuration['mean']),np.array(configuration['scale']))
    from tensorflow import keras
    model=keras.models.load_model(args.experiment/'model.keras')
    probabilities=model(features,training=False).numpy()[0]
    classes=configuration['classes']
    print(json.dumps({'video_id':video.video_id,'start_s':args.start,'end_s':args.start+2,
                      'predicted_class':classes[int(probabilities.argmax())],
                      'confidence':float(probabilities.max()),
                      'class_probabilities':dict(zip(classes,probabilities.astype(float).tolist())),
                      'note':'Probabilities are not calibrated; this model has no trained unknown class.'},indent=2))


if __name__=='__main__':main()
