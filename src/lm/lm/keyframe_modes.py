"""Shared policy-mode requirements for the manipulation keyframe library."""

from __future__ import annotations


# Every action in the pick-and-place library must expose the measured object
# pose to the policy, including the standing setup and final standing frames.
MANIPULATION_KEYFRAMES = frozenset(
    {
        "stand_before_pick",
        "crouch_to_pick",
        "stand_after_pick",
        "stand_before_place",
        "crouch_to_place",
        "stand_after_place",
    }
)
