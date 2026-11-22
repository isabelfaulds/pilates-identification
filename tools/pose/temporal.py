"""Causal, timestamp-based angle history; never mixes videos or segments."""
from collections import deque
import math

EPS = 1e-9


def seconds_label(value):
    return f'{value:g}'.replace('.', 'p') + 's'


class TemporalAngles:
    def __init__(self, angle_names, lag_seconds=(.1,.5,1.), mean_seconds=(.5,1.),
                 mean_lag_seconds=.5, max_gap_seconds=.1, lag_tolerance_seconds=.05):
        self.angle_names = tuple(angle_names)
        self.lags = tuple(lag_seconds)
        self.means = tuple(mean_seconds)
        self.mean_lag = mean_lag_seconds
        self.max_gap = max_gap_seconds
        self.tolerance = lag_tolerance_seconds
        settings = self.lags+self.means+(self.mean_lag,self.max_gap)
        if any(not math.isfinite(x) or x <= 0 for x in settings):
            raise ValueError('Temporal periods and maximum gap must be finite and positive')
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('Lag tolerance must be finite and nonnegative')
        if len(set(self.lags)) != len(self.lags) or len(set(self.means)) != len(self.means):
            raise ValueError('Temporal periods must be distinct')
        self.retention = max((0.,)+self.lags+tuple(w+self.mean_lag for w in self.means))+self.tolerance+self.max_gap
        self.states = {}
        self.prefixes = []
        for angle in self.angle_names:
            self.prefixes.extend(angle+'_lag_'+seconds_label(lag) for lag in self.lags)
            self.prefixes.extend(angle+'_mean_'+seconds_label(window) for window in self.means)
            self.prefixes.extend(angle+'_mean_'+seconds_label(window)+'_lag_'+seconds_label(self.mean_lag)
                                 for window in self.means)
        self.columns = [column for prefix in self.prefixes for column in (prefix+'_deg',prefix+'_valid')]
        self.columns.extend(column for angle in self.angle_names
                            for column in (angle+'_velocity_deg_per_s',angle+'_velocity_valid'))

    def past(self, history, target):
        for item in reversed(history):
            if item[0] <= target+EPS:
                return item if target-item[0] <= self.tolerance+EPS else None
        return None

    def update(self, video_id, timestamp, features, segment_id=None):
        if not math.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        state = self.states.get(video_id)
        if state and timestamp <= state['last_time']:
            raise ValueError('Timestamps must increase within each video')
        if (state is None or segment_id != state['segment'] or
                timestamp-state['last_time'] > self.max_gap+EPS):
            state = {'last_time':timestamp, 'segment':segment_id,
                     'histories':{name:deque() for name in self.angle_names}}
            self.states[video_id] = state
        state['last_time'] = timestamp
        result = {column:(0 if column.endswith('_valid') else '') for column in self.columns}

        def put(prefix, value):
            if value is not None and math.isfinite(value):
                result[prefix+'_deg'] = value
                result[prefix+'_valid'] = 1

        for name,history in state['histories'].items():
            value = features[name+'_deg']
            if not features[name+'_valid'] or value == '' or not math.isfinite(float(value)):
                history.clear()
                continue
            previous = history[-1] if history else None
            if previous is not None:
                velocity = (float(value)-previous[1])/(timestamp-previous[0])
                if math.isfinite(velocity):
                    result[name+'_velocity_deg_per_s'] = velocity
                    result[name+'_velocity_valid'] = 1
            current_means = {}
            history.append((timestamp,float(value),current_means))
            while len(history) > 1 and history[1][0] < timestamp-self.retention:
                history.popleft()
            for window in self.means:
                start = timestamp-window
                values = [point[1] for point in history if point[0] >= start-EPS]
                # Full warm-up period; missing/low-confidence points reset this history.
                mean = (math.fsum(values)/len(values) if history[0][0] <= start+EPS
                        and len(values) >= 2 else None)
                current_means[window] = mean
                put(name+'_mean_'+seconds_label(window),mean)
            for lag in self.lags:
                point = self.past(history,timestamp-lag)
                put(name+'_lag_'+seconds_label(lag),point[1] if point else None)
            point = self.past(history,timestamp-self.mean_lag)
            for window in self.means:
                put(name+'_mean_'+seconds_label(window)+'_lag_'+seconds_label(self.mean_lag),
                    point[2].get(window) if point else None)
        return result
