"""Compute the transcription statistics of Table 3 and the segment shares of Table 7.

Table 3  For each corpus: number of videos, total duration, transcribed words,
         words per minute of video, speech coverage (union of the segment
         intervals divided by the video duration), number of speech segments,
         mean, median, 90th and 95th percentile of segment length, and the share
         of segments in each duration band of the frame sampling rule
         (Equation 1): shorter than tau1, from tau1 to less than tau2, and tau2
         or longer.
Table 7  Share of segments that receive a single frame under the thresholds
         (tau1, tau2) = (6, 12), (8, 16) and (10, 20) s.

Input: a folder with one JSON file per video holding the Whisper output, either
a list of segments or an object with a "segments" list; each segment needs
"start", "end" (seconds) and "text". The duration of each video is read from a
"duration" field in the JSON file or, preferably, from a CSV file passed with
--durations (columns video_id,duration_s; video_id is the JSON file name without
extension). Videos in which no speech was detected must still appear, with an
empty segment list or in the durations file, so that they count toward the
total duration.

Usage:
    python transcript_statistics.py whisper_json_folder [--durations durations.csv]

The relative frame budget of Table 7 also counts the reference frames placed in
silent intervals, so it is computed by the sampling code of the pipeline, not here.
Requirements: numpy.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

SCHEDULES = [(6.0, 12.0), (8.0, 16.0), (10.0, 20.0)]


def frames_per_segment(d, tau1, tau2):
    """Step rule of Equation 1: one, two or three frames depending on duration d."""
    return 1 if d < tau1 else (2 if d < tau2 else 3)


def union_length(intervals):
    total, cur_start, cur_end = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def load(folder, durations_csv):
    durations = {}
    if durations_csv:
        with open(durations_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                durations[row["video_id"].strip()] = float(row["duration_s"])
    videos = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        vid = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        segs = obj if isinstance(obj, list) else obj.get("segments", [])
        if vid not in durations:
            if isinstance(obj, dict) and "duration" in obj:
                durations[vid] = float(obj["duration"])
            else:
                raise ValueError(f"no duration for {vid}; pass --durations")
        videos[vid] = [(float(s["start"]), float(s["end"]), s.get("text", "")) for s in segs]
    for vid in durations:  # videos listed in the durations file with no JSON file
        videos.setdefault(vid, [])
    return videos, durations


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--durations")
    args = ap.parse_args()
    videos, durations = load(args.folder, args.durations)

    total_s = sum(durations[v] for v in videos)
    lengths = np.array([e - s for segs in videos.values() for s, e, _ in segs])
    words = sum(len(t.split()) for segs in videos.values() for _, _, t in segs)
    covered = sum(union_length([(s, e) for s, e, _ in segs]) for segs in videos.values())

    print("Table 3")
    print(f"  Videos: {len(videos)} ({sum(1 for s in videos.values() if not s)} without detected speech)")
    print(f"  Total duration (h): {total_s / 3600:.2f}")
    print(f"  Transcribed words: {words}")
    print(f"  Words per minute of video: {words / (total_s / 60):.2f}")
    print(f"  Speech coverage (%): {100 * covered / total_s:.2f}")
    print(f"  Speech segments: {lengths.size}")
    if lengths.size:
        p90, p95 = np.percentile(lengths, [90, 95])
        print(f"  Mean / median segment length (s): {lengths.mean():.2f} / {np.median(lengths):.2f}")
        print(f"  90th / 95th percentile of segment length (s): {p90:.2f} / {p95:.2f}")
        for tau1, tau2 in SCHEDULES:
            n = np.array([frames_per_segment(d, tau1, tau2) for d in lengths])
            shares = [100 * np.mean(n == k) for k in (1, 2, 3)]
            tag = "Table 3 bands" if (tau1, tau2) == (8.0, 16.0) else "Table 7"
            print(f"  ({tau1:g} s, {tau2:g} s) segments with 1 / 2 / 3 frames (%): "
                  f"{shares[0]:.2f} / {shares[1]:.2f} / {shares[2]:.2f}   [{tag}]")


if __name__ == "__main__":
    main()
