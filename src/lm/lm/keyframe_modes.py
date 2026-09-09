"""Shared policy-mode requirements for the manipulation keyframe library."""

from __future__ import annotations


# Fixed VLM-node mapping, independent of model output and pickup distance.
# approach uses locomotion/no-object mode; all six pick/place actions expose
# the object, including stand_before_pick.
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


def source_keyframe_name(keyframe_name: str) -> str:
    """Approach reuses the pickup stand; no duplicate NPZ is required."""
    return "stand_before_pick" if keyframe_name == "approach" else keyframe_name
