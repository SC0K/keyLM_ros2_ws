"""Separate, freshness-checked mocap poses; never substitute the other object."""

import math
import time

TRACKED_OBJECT_DEFAULTS = {
    "mocap_object_selection": False,
    "tracked_box_pose_topic": "/mocap/box_pose",
    "tracked_bucket_pose_topic": "/mocap/bucket_pose",
    "tracked_object_timeout_sec": 1.0,
}


def validate_object_topics(box, bucket, active):
    if not all((box, bucket, active)) or len({box, bucket, active}) != 3:
        raise ValueError("Box, bucket, and selected-object pose topics must be distinct")


class TrackedObjects:
    def __init__(self, timeout=1.0):
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("tracked_object_timeout_sec must be positive")
        self.poses = {}

    def update(self, kind, message):
        p, q = message.pose.position, message.pose.orientation
        values = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
        if kind not in ("box", "bucket") or not all(math.isfinite(v) for v in values):
            return False
        if sum(v * v for v in values[3:]) < 1e-12:
            return False
        self.poses[kind] = (message, time.monotonic())
        return True

    def get(self, kind):
        entry = self.poses.get(kind)
        if entry is None or time.monotonic() - entry[1] > self.timeout:
            return None
        return entry[0]
