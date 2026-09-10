"""Shared policy-mode requirements for the manipulation keyframe library."""

from __future__ import annotations


# Fixed VLM-node mapping, independent of model output and pickup distance.
# approach uses locomotion/no-object mode; all six pick/place actions expose
# the object, including stand_before_pick.
MANIPULATION_PHASES = frozenset(
    {
        "stand_before_pick",
        "crouch_to_pick",
        "stand_after_pick",
        "stand_before_place",
        "crouch_to_place",
        "stand_after_place",
    }
)

PLANNER_KEYFRAMES = tuple(f"{phase}_{kind}" for kind in ("box", "bucket")
                         for phase in ("approach", *sorted(MANIPULATION_PHASES)))
MANIPULATION_KEYFRAMES = MANIPULATION_PHASES | frozenset(
    name for name in PLANNER_KEYFRAMES if not name.startswith("approach_"))


def keyframe_phase(name: str) -> str:
    for kind in ("box", "bucket"):
        if name.endswith("_" + kind):
            return name[:-(len(kind) + 1)]
    return name


def keyframe_object_type(name: str, default: str = "box") -> str:
    return next((kind for kind in ("box", "bucket") if name.endswith("_" + kind)), default)


def source_keyframe_name(keyframe_name: str) -> str:
    """Approach reuses the pickup stand; no duplicate NPZ is required."""
    phase = keyframe_phase(keyframe_name)
    source = "stand_before_pick" if phase == "approach" else phase
    return source if phase == keyframe_name else f"{source}_{keyframe_object_type(keyframe_name)}"
