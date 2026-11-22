"""Train a same-recording LSTM pilot with protected clean/noisy test snippets."""
import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

if __package__:
    from .features import ANGLE_JOINTS, FEATURE_COLUMNS, JOINTS, row_features
    from .temporal import TemporalAngles
else:
    from features import ANGLE_JOINTS, FEATURE_COLUMNS, JOINTS, row_features
    from temporal import TemporalAngles

ROOT = Path(__file__).resolve().parents[2]
FPS = 30
WINDOW_SECONDS = 2.
GUARD_SECONDS = 1.5
COLUMNS = FEATURE_COLUMNS+TemporalAngles(ANGLE_JOINTS).columns
VALUE_COLUMNS = [name for name in COLUMNS if not name.endswith('_valid')]
VALID_COLUMNS = [name.removesuffix('_deg_per_s')+'_valid' if name.endswith('_deg_per_s')
                 else name.removesuffix('_deg')+'_valid' for name in VALUE_COLUMNS]


@dataclass
class Video:
    video_id: str
    name: str
    label: str
    rows: list
    timestamps: np.ndarray
    angles: list


def load_videos(source, mapping):
    required = {'video_id','video_name','exercise_name','timestamp_s','frame_index','pose_detected'}
    required.update(f'world_{joint}_{axis}' for joint in JOINTS for axis in ('x','y','z'))
    required.update(f'{space}_{joint}_{score}' for joint in JOINTS for space in ('image','world')
                    for score in ('visibility','presence'))
    grouped = {}
    with source.open(newline='') as file:
        reader = csv.DictReader(file)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Expected combined raw landmark CSV')
        for record in reader:
            label = mapping.get(record['exercise_name']) if mapping is not None else ''
            if label is None:
                raise ValueError(f'No class mapping for {record["exercise_name"]}')
            row = {key:record[key] for key in required}
            grouped.setdefault(row['video_id'],[]).append(row)
    videos = {}
    for video_id,rows in sorted(grouped.items()):
        timestamps = np.array([float(row['timestamp_s']) for row in rows])
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f'Non-chronological frames in {video_id}')
        if len({(row['exercise_name'],row['video_name']) for row in rows}) != 1:
            raise ValueError(f'Video identity changed in {video_id}')
        videos[video_id] = Video(video_id,rows[0]['video_name'],mapping[rows[0]['exercise_name']] if mapping is not None else '',
                                 rows,timestamps,[row_features(row) for row in rows])
    if not videos:
        raise ValueError('No videos found')
    return videos


def sampled_indices(timestamps, start, seconds=WINDOW_SECONDS, fps=FPS):
    grid = start+np.arange(round(seconds*fps))/fps
    indices = np.searchsorted(timestamps,grid+1e-9,side='right')-1
    if np.any(indices < 0):
        return None
    age = grid-timestamps[indices]
    if len(np.unique(indices)) != len(indices) or np.any(age > .05+1e-9) or np.any(age < -1e-8):
        return None
    return indices


def intersects(start, end, other_start, other_end):
    return start < other_end-1e-9 and end > other_start+1e-9


def split_candidates(starts, seed, seconds=WINDOW_SECONDS, guard=GUARD_SECONDS,
                     test_count=2, val_count=1):
    """Reserve blocks first, then remove all intersecting/buffered train windows."""
    rng = np.random.default_rng(seed)
    starts = np.array(starts,dtype=float)
    minimum_train = max(2,min(10,len(starts)//3))
    for _ in range(200):
        held = []
        for start in rng.permutation(starts):
            if not any(intersects(start-guard,start+seconds+guard,h,h+seconds) for h in held):
                held.append(float(start))
            if len(held) == test_count+val_count:
                break
        if len(held) != test_count+val_count:
            continue
        train = [float(start) for start in starts
                 if not any(intersects(start,start+seconds,h-guard,h+seconds+guard) for h in held)]
        if len(train) >= minimum_train:
            return {'test':held[:test_count], 'validation':held[test_count:], 'train':train}
    raise ValueError('Not enough usable duration for protected train/validation/test chunks')


def make_manifest(videos, seed, trim_start=10., trim_end=5., min_valid=.5):
    manifest = []
    for video in videos.values():
        # A provisional trim, not a claim that all retained frames were annotated.
        stop = video.timestamps[-1]+1/FPS-trim_end
        candidates = []
        for start in np.arange(trim_start,stop-WINDOW_SECONDS+1e-9,.5):
            indices = sampled_indices(video.timestamps,float(start))
            if indices is None:
                continue
            valid = np.array([[video.angles[i][name+'_valid'] for name in ANGLE_JOINTS]
                              for i in indices],dtype=bool)
            if valid.mean() >= min_valid:
                candidates.append(float(start))
        video_seed = seed+int(hashlib.sha256(video.video_id.encode()).hexdigest()[:8],16)
        partitions = split_candidates(candidates,video_seed)
        for split,starts in partitions.items():
            for start in sorted(starts):
                indices = sampled_indices(video.timestamps,start)
                manifest.append({'window_id':f'{video.video_id}_{start:.3f}',
                                 'video_id':video.video_id,'video_name':video.name,'class_name':video.label,
                                 'split':split,'start_s':start,'end_s':start+WINDOW_SECONDS,
                                 'source_frame_indices':[int(video.rows[i]['frame_index']) for i in indices]})
    return manifest


def assert_protected_splits(manifest, guard=GUARD_SECONDS):
    for i,left in enumerate(manifest):
        for right in manifest[i+1:]:
            if left['video_id'] != right['video_id'] or left['split'] == right['split']:
                continue
            if intersects(left['start_s'],left['end_s'],right['start_s']-guard,right['end_s']+guard):
                raise ValueError('Split windows or guard intervals overlap')
            if set(left['source_frame_indices']).intersection(right['source_frame_indices']):
                raise ValueError('Source frames shared between splits')


def window_features(video, indices, noise_sigma=0., seed=42):
    temporal = TemporalAngles(ANGLE_JOINTS)
    rng = np.random.default_rng(seed)
    values = []
    for index in indices:
        row = video.rows[index]
        if noise_sigma:
            row = dict(row)
            # Add coordinate noise, then recalculate dependent angle/temporal features.
            for joint in JOINTS:
                for axis in ('x','y','z'):
                    key = f'world_{joint}_{axis}'
                    if row[key] != '' and math.isfinite(float(row[key])):
                        row[key] = str(float(row[key])+rng.normal(0.,noise_sigma))
            base = row_features(row)
        else:
            base = dict(video.angles[index])
        base.update(temporal.update(video.video_id,float(row['timestamp_s']),base))
        values.append([float(base[column]) if base[valid] and base[column] != '' else np.nan
                       for column,valid in zip(VALUE_COLUMNS,VALID_COLUMNS)])
    return np.array(values,dtype=np.float32)


def build_arrays(videos,manifest,classes,split,noise_sigma=0.,seed=42):
    records = [item for item in manifest if item['split'] == split]
    x,y = [],[]
    for item in records:
        video = videos[item['video_id']]
        indices = sampled_indices(video.timestamps,item['start_s'])
        noise_seed = seed+int(hashlib.sha256(item['window_id'].encode()).hexdigest()[:8],16)
        x.append(window_features(video,indices,noise_sigma,noise_seed))
        y.append(classes.index(video.label))
    return np.stack(x),np.array(y,dtype=np.int32),records


def fit_scaler(values):
    mask = np.isfinite(values)
    counts = mask.sum(axis=(0,1))
    mean = np.where(mask,values,0).sum(axis=(0,1))/np.maximum(counts,1)
    variance = np.where(mask,(values-mean)**2,0).sum(axis=(0,1))/np.maximum(counts,1)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.
    return mean,scale,counts


def transform(values,mean,scale):
    valid = np.isfinite(values)
    normalized = np.where(valid,(values-mean)/scale,0.)
    # Flags distinguish imputed/missing values from a genuine normalized zero.
    return np.concatenate([normalized,valid.astype(np.float32)],axis=-1).astype(np.float32)


def train(args):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    configuration = json.loads(args.classes.read_text())
    classes = configuration['classes']
    videos = load_videos(args.input,configuration['exercise_to_class'])
    manifest = make_manifest(videos,args.seed,args.trim_start,args.trim_end,args.min_valid)
    assert_protected_splits(manifest)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'split_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    arrays = {}
    labels = {}
    records = {}
    for split in ('train','validation','test'):
        arrays[split],labels[split],records[split] = build_arrays(videos,manifest,classes,split,seed=args.seed)
        if set(labels[split]) != set(range(len(classes))):
            raise ValueError(f'{split} does not contain every class')
        print(f'{split}: {len(labels[split])} snippets',flush=True)
    noisy,ny,nrecords = build_arrays(videos,manifest,classes,'test',args.noise_sigma,args.seed)
    np.testing.assert_array_equal(ny,labels['test'])
    assert nrecords == records['test']
    mean,scale,counts = fit_scaler(arrays['train'])
    x = {key:transform(value,mean,scale) for key,value in arrays.items()}
    noisy = transform(noisy,mean,scale)
    def dataset(features,targets,shuffle=False):
        data = tf.data.Dataset.from_tensor_slices((features,targets))
        if shuffle:
            data = data.shuffle(len(features),seed=args.seed)
        options = tf.data.Options(); options.threading.private_threadpool_size=1
        options.threading.max_intra_op_parallelism=1
        return data.batch(16).with_options(options)
    model = keras.Sequential([keras.layers.Input(shape=x['train'].shape[1:]),
                              keras.layers.LSTM(32),
                              keras.layers.Dense(len(classes),activation='softmax')])
    model.compile(optimizer=keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    class_counts = np.bincount(labels['train'],minlength=len(classes))
    weights = {i:len(labels['train'])/(len(classes)*count) for i,count in enumerate(class_counts)}
    history = model.fit(dataset(x['train'],labels['train'],True),
                        validation_data=dataset(x['validation'],labels['validation']),
                        epochs=args.epochs,verbose=2,class_weight=weights,shuffle=False,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True)])
    metrics = {}
    for condition,inputs in [('clean',x['test']),('noisy',noisy)]:
        probabilities = model(inputs,training=False).numpy()
        predicted = probabilities.argmax(axis=1)
        metrics[condition] = {'accuracy':float(accuracy_score(labels['test'],predicted)),
                              'macro_f1':float(f1_score(labels['test'],predicted,labels=list(range(len(classes))),average='macro',zero_division=0)),
                              'confusion_matrix':confusion_matrix(labels['test'],predicted,labels=list(range(len(classes)))).tolist(),
                              'classification_report':classification_report(labels['test'],predicted,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)}
        with (args.output/f'{condition}_test_predictions.csv').open('w',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['window_id','video_name','start_s','end_s','actual','predicted','confidence']+
                            ['p_'+name for name in classes])
            for item,prediction,prob in zip(records['test'],predicted,probabilities):
                writer.writerow([item['window_id'],item['video_name'],item['start_s'],item['end_s'],
                                 item['class_name'],classes[prediction],float(prob.max())]+prob.tolist())
    majority = int(class_counts.argmax())
    metrics['majority_baseline_accuracy'] = float(np.mean(labels['test']==majority))
    metrics['classes'] = classes
    metrics['split_counts'] = {split:len(y) for split,y in labels.items()}
    metrics['training_class_counts'] = dict(zip(classes,class_counts.tolist()))
    metrics['epochs_run'] = len(history.history['loss'])
    metrics['best_epoch'] = int(np.argmin(history.history['val_loss'])+1)
    metrics['limitations'] = ['Same-recording pilot; not unseen-video or unseen-person evaluation.',
                              'Video-title labels with provisional edge trimming; exercise intervals need manual review.',
                              'Noisy test is a paired robustness check, not an independent new test set.']
    model.save(args.output/'model.keras')
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (args.output/'history.json').write_text(json.dumps(history.history,indent=2)+'\n')
    (args.output/'preprocessing.json').write_text(json.dumps({
        'value_columns':VALUE_COLUMNS,'validity_columns':VALID_COLUMNS,'mean':mean.tolist(),
        'scale':scale.tolist(),'training_observed_counts':counts.tolist(),
        'missing_values':'training mean imputation, zero after scaling; append finite-value flags',
        'classes':classes,'window_seconds':WINDOW_SECONDS,'fps':FPS,
        'input_shape':list(x['train'].shape[1:]),'seed':args.seed,
        'noise_sigma_world_units':args.noise_sigma,'training_augmentation':False,
        'trim_start_seconds':args.trim_start,'trim_end_seconds':args.trim_end,
        'split_guard_seconds':GUARD_SECONDS,'minimum_valid_angle_fraction':args.min_valid,
        'temporal_features':'recomputed from scratch inside each 2-second snippet; default TemporalAngles settings',
        'tensorflow_version':tf.__version__,
        'minimum_training_windows_per_video':'max(2, min(10, floor(candidate_count/3)))',
        'temporal_parameters':{'lag_seconds':[.1,.5,1.], 'mean_seconds':[.5,1.], 'mean_lag_seconds':.5,'max_gap_seconds':.1,'lag_tolerance_seconds':.05},
        'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({key:metrics[key] for key in ['clean','noisy','majority_baseline_accuracy','split_counts']},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/processed/landmarks.csv')
    parser.add_argument('--classes',type=Path,default=ROOT/'config/pose_classes.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/experiments/lstm_pilot')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--noise-sigma',type=float,default=.005)
    parser.add_argument('--trim-start',type=float,default=10.)
    parser.add_argument('--trim-end',type=float,default=5.)
    parser.add_argument('--min-valid',type=float,default=.5)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output folder is not empty; choose a fresh --output to preserve prior experiments')
    if args.epochs <= 0 or not 0 <= args.min_valid <= 1 or not math.isfinite(args.noise_sigma) or args.noise_sigma < 0:
        parser.error('Invalid training parameters')
    if any(not math.isfinite(value) or value < 0 for value in (args.trim_start,args.trim_end)):
        parser.error('Trimming must be finite and nonnegative')
    train(args)


if __name__ == '__main__':
    main()
