"""Score normalization rule sets on M2's saved replay pairs, offline.

    python scripts/agentic_rule_check.py --pairs /root/m2-pairs.jsonl --out rules.json

A rule set is adoptable when every honest episode whose diff matched keeps at
least one observation of margin under ``allowed_mismatches`` (spec §7 M2).
A tightened rule takes the place of the loose rule it replaces, so a candidate
set is scored in the order the normalizer applies it.
"""

import argparse
import json
import re

from reliquary.corpus.replay_compare import _RULES, allowed_mismatches

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
# The four tightened replacements, keyed by the loose rule they replace.
TIGHT = {
    # Only CPython reprs ("<Foo object at 0x7f..>"), never a hex literal in a file.
    "addr": (re.compile(r"(?<=at )0x[0-9a-f]{6,}\b"), "<addr>"),
    # Only an `ls -l` line's mtime column: mode, links, owner, group, size, then the date.
    "mtime": (re.compile(r"(?m)^([-dlcbps][-rwxsStT]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+\d+\s+)"
                         + _MONTH + r" +\d{1,2} +\d{2}:\d{2}"), r"\1<mtime>"),
    # Durations that stand alone ("0.75s", "12 s", "3 ms"), never inside an identifier.
    "duration": (re.compile(r"(?<![\w.])\d+(?:\.\d+)? ?(?:s|ms|secs?|seconds?)(?![\w.])"), "<dur>"),
    # A full hash only on a `git log`/`git show` header line.
    "commit": (re.compile(r"(?m)^commit [0-9a-f]{40}\b"), "commit <commit>"),
}
# Order in which a combined subset is grown (spec Task 2, Step 5).
ORDER = ("addr", "commit", "mtime", "duration")
# Which current rules each tightened one replaces, by their pattern text.
REPLACES = {
    "addr": {r"\b0x[0-9a-f]{6,}\b"},
    "mtime": {r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +\d{2}:\d{2}\b"},
    "duration": {r"\b\d+(\.\d+)?s\b", r"\b\d+(\.\d+)? ?(ms|seconds?|secs?)\b"},
    "commit": {r"(?<=commit )[0-9a-f]{40}\b"},
}


def rules_with(chosen: set[str]):
    """The loose pipeline with each chosen rule swapped in place (same order as the normalizer).

    The two loose duration rules collapse into the one tightened rule, at the
    position of the first.
    """
    out, placed = [], set()
    for pattern, replacement in _RULES:
        owner = next((n for n in chosen if pattern.pattern in REPLACES[n]), None)
        if owner is None:
            out.append((pattern, replacement))
        elif owner not in placed:
            placed.add(owner)
            out.append(TIGHT[owner])
    missing = chosen - placed
    assert not missing, f"loose rule not found in _RULES for {missing}"
    return out


def apply(text, rules):
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def score(episodes, rules):
    spares = []
    for ep in episodes:
        if not ep["diff_equal"]:
            continue
        pairs = ep["pairs"]
        mismatched = sum(1 for rec, rep in pairs if rep is None or apply(rec, rules) != apply(rep, rules))
        spares.append(allowed_mismatches(len(pairs)) - mismatched)
    return {"episodes": len(spares), "min_spare": min(spares), "mismatched_total":
            sum(allowed_mismatches(len(e["pairs"])) for e in episodes if e["diff_equal"]) - sum(spares)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    episodes = [json.loads(line) for line in open(args.pairs)]
    result = {"current": score(episodes, rules_with(set()))}
    for name in ORDER:
        result[name] = score(episodes, rules_with({name}))
    adoptable = [name for name in ORDER if result[name]["min_spare"] >= 1]
    result["adoptable"] = adoptable
    result["all_adoptable_together"] = score(episodes, rules_with(set(adoptable))) if adoptable else None
    chosen: list[str] = []
    for name in adoptable:  # largest subset, grown in ORDER, that holds together
        if score(episodes, rules_with(set(chosen + [name])))["min_spare"] >= 1:
            chosen.append(name)
    result["adopted"] = chosen
    result["adopted_score"] = score(episodes, rules_with(set(chosen))) if chosen else None
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
