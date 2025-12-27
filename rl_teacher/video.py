"""Video recording and export utilities for RL-Teacher."""
import os
import os.path as osp
import subprocess

import imageio
import numpy as np


class SegmentVideoRecorder:
    """Records videos of agent behavior at checkpoints."""

    def __init__(self, predictor, env, save_dir, checkpoint_interval=500):
        self.predictor = predictor
        self.env = env
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = save_dir

        self._num_paths_seen = 0
        self._counter = 0

    def path_callback(self, path):
        """Called after each episode path."""
        if self._num_paths_seen % self.checkpoint_interval == 0:
            fname = "%s/run_%s_%s.mp4" % (self.save_dir, self._num_paths_seen, self._counter)
            print("Saving video of run %s_%s to %s" % (self._num_paths_seen, self._counter, fname))
            write_segment_to_video(path, fname, self.env)
        self._num_paths_seen += 1

        self.predictor.path_callback(path)

    def predict_reward(self, path):
        """Predict reward using the wrapped predictor."""
        return self.predictor.predict_reward(path)


def write_segment_to_video(segment, fname, env):
    """Write a segment to a video file."""
    os.makedirs(osp.dirname(fname), exist_ok=True)

    frames = []
    for human_obs in segment["human_obs"]:
        frame = env.render_full_obs(human_obs)
        if frame is not None:
            frames.append(frame)

    # Add a short pause at the end
    if frames:
        for _ in range(int(env.fps * 0.2)):
            frames.append(frames[-1])

    export_video(frames, fname, fps=env.fps)


def export_video(frames, fname, fps=10):
    """Export frames to a video file using imageio."""
    assert fname.endswith(".mp4"), "Name requires .mp4 suffix"
    os.makedirs(osp.dirname(fname), exist_ok=True)

    if not frames:
        print(f"Warning: No frames to export for {fname}")
        return

    # Convert frames to proper format if needed
    processed_frames = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # Ensure uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            processed_frames.append(frame)

    if processed_frames:
        imageio.mimwrite(fname, processed_frames, fps=fps, codec="libx264", quality=8)


def upload_to_gcs(local_path, gcs_path):
    """Upload a file to Google Cloud Storage."""
    assert osp.isfile(local_path), "%s must be a file" % local_path
    assert gcs_path.startswith("gs://"), "%s must start with gs://" % gcs_path

    print("Copying media to %s in a background process" % gcs_path)
    subprocess.check_call(["gsutil", "cp", local_path, gcs_path])
