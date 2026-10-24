"""Bounded rewards for reviewed image grounding and football events."""

from __future__ import annotations

import math

from reliquary.shared.strict_json import strict_json_loads

# Football weights and decay match Score's private soccer_action scorer at
# 60b328cbaabe63024f0e034204a0f6deb0d3154f; confidence is diagnostic only.
ACTION_CONFIGS = {
    "pass": (1.0, 0.0, 1.0),
    "pass_received": (1.4, 0.0, 1.0),
    "recovery": (1.5, 0.0, 1.5),
    "tackle": (2.5, 0.1, 1.5),
    "interception": (2.8, 0.5, 2.0),
    "ball_out_of_play": (2.9, 0.5, 2.0),
    "clearance": (3.1, 0.5, 2.0),
    "take_on": (3.2, 0.5, 2.0),
    "substitution": (4.2, 0.5, 2.0),
    "block": (4.2, 0.5, 2.0),
    "aerial_duel": (4.3, 0.5, 2.0),
    "shot": (4.7, 0.5, 2.0),
    "save": (7.3, 0.5, 2.0),
    "foul": (7.7, 0.5, 2.5),
    "goal": (10.9, 0.5, 3.0),
}


def validate_target(target: object, task: str, duration_us: int) -> dict:
    """Reject ambiguous labels; coordinates and football frames are canonical."""
    if not isinstance(target, dict) or set(target) != {"type", "items"}:
        raise ValueError("target must contain exactly type and items")
    expected = {"football_events": "soccer_action", "image_grounding": "grounding"}
    if task not in expected or target["type"] != expected[task]:
        raise ValueError("target type does not match task")
    items = target["items"]
    limit = 256 if task == "football_events" else 64
    if not isinstance(items, list) or len(items) > limit:
        raise ValueError("target items exceed the task bound")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("target item must be an object")
        if task == "football_events":
            if set(item) != {"frame", "action", "confidence"}:
                raise ValueError("event requires frame, action and confidence")
            frame = item["frame"]
            if (type(frame) not in (int, float) or not 0 <= frame < duration_us * 25 / 1_000_000
                    or int(frame) != frame):
                raise ValueError("event frame must be inside the clip at 25 FPS")
            item["frame"] = int(frame)
            if not isinstance(item["action"], str) or item["action"] not in ACTION_CONFIGS:
                raise ValueError("unknown football action")
            confidence = item["confidence"]
            if (type(confidence) not in (int, float) or not 0 <= confidence <= 1
                    or not math.isfinite(confidence)):
                raise ValueError("confidence must be finite and between zero and one")
        else:
            if set(item) != {"label", "box"}:
                raise ValueError("grounding item requires label and box")
            label, box = item["label"], item["box"]
            if (not isinstance(label, str) or not 1 <= len(label) <= 64
                    or label.strip() != label):
                raise ValueError("label must be canonical text of at most 64 characters")
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(x) not in (int, float) or not 0 <= x <= 1000
                           or int(x) != x for x in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("box must be nonempty normalized integer xyxy in [0,1000]")
            item["box"] = [int(x) for x in box]
    return target


def football_score(predictions: list[dict], target: list[dict]) -> float:
    """Score-compatible per-clip metric, including zero for empty ground truth."""
    if not target:
        return 0.0
    total = sum(ACTION_CONFIGS[item["action"]][0] for item in target)
    used: set[int] = set()
    earned = 0.0
    for prediction in sorted(predictions, key=lambda item: item["frame"]):
        weight, minimum, tolerance = ACTION_CONFIGS[prediction["action"]]
        best_index, best_decay = None, 0.0
        for index, truth in enumerate(target):
            if index in used or truth["action"] != prediction["action"]:
                continue
            distance = abs(prediction["frame"] / 25 - truth["frame"] / 25)
            decay = 1 - distance / tolerance * (1 - minimum) if distance <= tolerance else 0
            if decay > best_decay:
                best_index, best_decay = index, decay
        if best_index is None:
            earned -= weight
        else:
            used.add(best_index)
            earned += weight * best_decay
    return max(0.0, min(1.0, earned / total))


def _iou(left: list[int], right: list[int]) -> float:
    intersection = (max(0, min(left[2], right[2]) - max(left[0], right[0]))
                    * max(0, min(left[3], right[3]) - max(left[1], right[1])))
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return intersection / union


def grounding_score(predictions: list[dict], target: list[dict]) -> float:
    """One-to-one class-aware IoU, penalizing missing and extra boxes.

    This service metric is not Score's public polygon/AP evaluator.
    """
    if not target:
        return 0.0
    pairs = sorted(
        ((_iou(prediction["box"], truth["box"]), i, j)
         for i, prediction in enumerate(predictions)
         for j, truth in enumerate(target) if prediction["label"] == truth["label"]),
        reverse=True,
    )
    used_predictions, used_targets = set(), set()
    earned = 0.0
    for overlap, i, j in pairs:
        if i not in used_predictions and j not in used_targets:
            used_predictions.add(i)
            used_targets.add(j)
            earned += overlap
    return earned / max(len(target), len(predictions))


def reward(example: dict, completion: str, training: bool = False) -> float:
    """Invalid completions score zero; invalid/unreviewed annotations fail closed."""
    task = example["task"]
    target = validate_target(example["target"], task, example["media"]["duration_us"])
    if (example.get("annotation_complete") is not True
            or type(example.get("is_verified_negative")) is not bool
            or example["is_verified_negative"] != (not target["items"])):
        raise ValueError("reward requires complete annotations and verified empty labels")
    if not isinstance(completion, str):
        return 0.0
    try:
        if len(completion.encode("utf-8")) > 65_536:
            return 0.0
    except UnicodeEncodeError:
        return 0.0
    try:
        prediction = validate_target(
            strict_json_loads(completion), task, example["media"]["duration_us"])
    except (ValueError, TypeError, RecursionError):
        return 0.0
    if training and example["is_verified_negative"]:
        return float(not prediction["items"])
    scorer = football_score if task == "football_events" else grounding_score
    return scorer(prediction["items"], target["items"])
