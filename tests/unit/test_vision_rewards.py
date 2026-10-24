from __future__ import annotations

import json

import pytest

from reliquary.vision.rewards import reward


def football(target: list[dict] | None = None) -> dict:
    return {
        "task": "football_events", "annotation_complete": True,
        "is_verified_negative": target == [],
        "media": {"duration_us": 30_000_000},
        "target": {"type": "soccer_action", "items": target if target is not None else [event(250)]},
    }


def event(frame: int, action: str = "pass", confidence: float = 1.0) -> dict:
    return {"frame": frame, "action": action, "confidence": confidence}


def response(items: list[dict], kind: str = "soccer_action") -> str:
    return json.dumps({"type": kind, "items": items})


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ([event(250)], 1.0),
        ([event(262)], 0.52),
        ([event(275)], 0.0),
        ([event(250), event(250)], 0.0),
        ([event(250, "shot")], 0.0),
        ([], 0.0),
        ([event(250, confidence=0.01)], 1.0),
        ([event(300)], 0.0),
    ],
)
def test_football_matches_score_metric_fixtures(prediction, expected):
    assert reward(football(), response(prediction)) == pytest.approx(expected)


def test_goal_decay_and_weighted_false_positive():
    assert reward(football([event(250, "goal")]), response([event(275, "goal")])) == pytest.approx(5 / 6)
    assert reward(football([event(250), event(500, "goal")]),
                  response([event(250), event(500, "goal"), event(510)])) == pytest.approx(10.9 / 11.9)


def test_verified_negative_training_reward_does_not_change_official_metric():
    example = football([])
    assert reward(example, response([])) == 0
    assert reward(example, response([event(250)])) == 0
    assert reward(example, response([]), training=True) == 1
    assert reward(example, response([event(250)]), training=True) == 0
    example["annotation_complete"] = False
    with pytest.raises(ValueError, match="complete annotations"):
        reward(example, response([]), training=True)


@pytest.mark.parametrize("completion", [
    '{"type":"soccer_action","items":NaN}',
    '{"type":"soccer_action","items":[],"items":[]}',
    "```json\n{\"type\":\"soccer_action\",\"items\":[]}\n```",
    "[]" , "null", "\ud800", "x" * 65_537,
    response([event(-1)]),
    response([event(750)]),
    response([event(1, confidence=float("nan"))]),
    response([event(1, action="unknown")]),
])
def test_invalid_completion_never_earns_negative_abstention_reward(completion):
    assert reward(football([]), completion, training=True) == 0


def test_grounding_iou_is_class_aware_and_penalizes_extra_boxes():
    target = {"label": "ball", "box": [0, 0, 1000, 1000]}
    example = {
        "task": "image_grounding", "annotation_complete": True,
        "is_verified_negative": False, "media": {"duration_us": 0},
        "target": {"type": "grounding", "items": [target]},
    }
    assert reward(example, response([target], "grounding")) == 1
    assert reward(example, response([target, target], "grounding")) == 0.5
    assert reward(example, response([dict(target, label="player")], "grounding")) == 0
    assert reward(example, response([dict(target, box=[0, 0, 500, 1000])], "grounding")) == 0.5
    assert reward(example, response([dict(target, box=[0, 0, 0, 0])], "grounding")) == 0
