#!/usr/bin/env python3
import os
import time
import json
import math
from datetime import datetime
from statistics import mean

# Simple learning progress estimator.
# It reads a metrics file (CSV or JSON lines) containing a timestamp and a numeric metric
# and estimates time to reach the next stage and time until the metric reaches a final goal.

METRICS_FILE = os.environ.get('METRICS_FILE', 'training_metrics.csv')
METRIC_COLUMN = os.environ.get('METRIC_COLUMN', 'value')
WINDOW = int(os.environ.get('ESTIMATOR_WINDOW', '20'))
GOAL_VALUE = os.environ.get('GOAL_VALUE', '')  # numeric string
GOAL_DIRECTION = os.environ.get('GOAL_DIRECTION', 'down')  # 'down' (loss) or 'up' (accuracy)
STAGE_THRESHOLDS_JSON = os.environ.get('STAGE_THRESHOLDS', '')  # json e.g. '{"stage1":0.5, "stage2":0.2}'
OUTPUT_FILE = os.environ.get('ESTIMATOR_OUTPUT', 'estimator_status.json')

def parse_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # Accept either: timestamp,value  or value  (timestamp optional)
            if len(parts) == 1:
                try:
                    val = float(parts[0])
                    rows.append((None, val))
                except Exception:
                    continue
            else:
                ts = parts[0]
                try:
                    val = float(parts[1])
                except Exception:
                    continue
                try:
                    # try parse timestamp as unix or ISO
                    if ts.isdigit():
                        t = int(ts)
                    else:
                        t = int(datetime.fromisoformat(ts).timestamp())
                except Exception:
                    t = int(time.time())
                rows.append((t, val))
    return rows

def linear_time_estimate(points, goal, direction='down'):
    # points: list of (t_seconds, value) with t increasing
    if len(points) < 2:
        return None
    # convert to (t, v) float arrays
    ts = [float(p[0] if p[0] is not None else i) for i, p in enumerate(points)]
    vs = [float(p[1]) for p in points]
    # fit linear regression v = a * t + b
    n = len(ts)
    mean_t = mean(ts)
    mean_v = mean(vs)
    num = sum((ts[i]-mean_t)*(vs[i]-mean_v) for i in range(n))
    den = sum((ts[i]-mean_t)**2 for i in range(n))
    if den == 0:
        return None
    a = num/den
    b = mean_v - a*mean_t
    # depending on direction, goal should be approached
    # if a==0 then no progress
    if a == 0:
        return None
    # compute t_goal = (goal - b) / a
    t_goal = (goal - b)/a
    now = time.time()
    # If estimate is in the past, return 0
    eta = t_goal - now
    if eta < 0:
        eta = 0
    return max(0.0, eta)

def human_readable(seconds):
    if seconds is None:
        return 'unknown'
    seconds = int(seconds)
    if seconds <= 0:
        return '0s'
    parts = []
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    return ' '.join(parts)

def load_stage_thresholds(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def main():
    rows = parse_csv(METRICS_FILE)
    status = {
        'timestamp': int(time.time()),
        'metric_points': len(rows),
        'next_stage_eta_seconds': None,
        'assistant_ready_eta_seconds': None,
        'next_stage': None,
        'assistant_ready_goal': None
    }

    if not rows:
        print(f"No metric data found in {METRICS_FILE}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(status, f)
        return

    # sort by timestamp if available
    rows_sorted = sorted(rows, key=lambda x: x[0] if x[0] is not None else 0)
    window = max(2, min(WINDOW, len(rows_sorted)))
    recent = rows_sorted[-window:]

    stages = load_stage_thresholds(STAGE_THRESHOLDS_JSON)
    # determine current metric value
    current_t, current_v = recent[-1]
    # guess next stage: the next threshold not yet reached
    next_stage = None
    next_threshold = None
    # assume thresholds are ordered by importance of lower/higher
    for name, thresh in stages.items():
        try:
            thresh_v = float(thresh)
        except Exception:
            continue
        if GOAL_DIRECTION == 'down':
            # lower is better
            if current_v > thresh_v:
                next_stage = name
                next_threshold = thresh_v
                break
        else:
            if current_v < thresh_v:
                next_stage = name
                next_threshold = thresh_v
                break

    status['next_stage'] = next_stage

    # estimate time to next stage
    if next_threshold is not None:
        eta_next = linear_time_estimate(recent, float(next_threshold), GOAL_DIRECTION)
        status['next_stage_eta_seconds'] = eta_next

    # estimate time to assistant ready if GOAL_VALUE set
    if GOAL_VALUE:
        try:
            goal = float(GOAL_VALUE)
            eta_ready = linear_time_estimate(recent, goal, GOAL_DIRECTION)
            status['assistant_ready_eta_seconds'] = eta_ready
            status['assistant_ready_goal'] = goal
        except Exception:
            pass

    # add human readable
    status['next_stage_eta'] = human_readable(status['next_stage_eta_seconds'])
    status['assistant_ready_eta'] = human_readable(status['assistant_ready_eta_seconds'])
    status['current_value'] = current_v

    # write to output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(status, f, indent=2)

    print(json.dumps(status, indent=2))

if __name__ == '__main__':
    main()