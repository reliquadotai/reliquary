"""A two-world EnvScaler corpus small enough to read, shaped like the real one.

Shared by the envscaler_tools_v2 tests. The real corpus is fetched, not
vendored, so behaviour is pinned here against worlds written to exercise it:
one honest shop whose clock, identifiers, random draws and set order would
each break a byte-for-byte replay if left alone, and one hostile world whose
every method reaches for the host.
"""

from __future__ import annotations

import json
from pathlib import Path

from reliquary.environment.agentic.envs.envscaler_tools_v2.corpus import (
    Corpus,
    build_corpus,
)

SHOP_SOURCE = '''
import time
import uuid
import random
from datetime import datetime


class Shop:
    def __init__(self):
        self.items = {}
        self.opened_at = time.time()

    def add_item(self, name, qty=1):
        """Add an item; every source of nondeterminism a world reaches for."""
        record = {
            "id": str(uuid.uuid4()),
            "qty": qty,
            "at": datetime.now().isoformat(),
            "lucky": random.randint(0, 10 ** 9),
            "tags": {"fresh", "new", name, "aisle-" + str(qty)},
        }
        self.items[name] = record
        return {"success": True, "data": record}

    def list_items(self):
        return {"success": True, "data": sorted(self.items)}

    def explode(self):
        raise ValueError("boom")
'''

HOSTILE_SOURCE = '''
import os
import socket


class Hostile:
    def __init__(self):
        self.notes = {}

    def write_file(self, path):
        with open(path, "w") as handle:
            handle.write("owned")
        return {"success": True}

    def read_file(self, path):
        with open(path) as handle:
            return {"success": True, "data": handle.read()}

    def connect(self, host, port):
        connection = socket.create_connection((host, int(port)), timeout=2)
        connection.close()
        return {"success": True}

    def spawn(self, command):
        return {"success": True, "status": os.system(command)}

    def fork(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        return {"success": True, "pid": pid}

    def spin(self):
        while True:
            pass

    def hog(self):
        block = bytearray(8 << 30)
        return {"success": True, "size": len(block)}

    def environment(self):
        return {"success": True, "data": dict(os.environ)}
'''


def _tool(tool: str, **properties: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool,
            "description": f"{tool} tool",
            "parameters": {
                "type": "object",
                "properties": {key: {"type": kind} for key, kind in properties.items()},
                "required": list(properties),
            },
        },
    }


def _world(env_id: str, class_name: str, source: str, tools: list[dict]) -> dict:
    return {
        "env_id": env_id,
        "env_class_name": class_name,
        "env_class_code": source,
        "environment_introduction": f"You operate the {class_name} system.",
        "environment_summary": class_name,
        "constraints_rules": ["Quantities are positive."],
        "tools": tools,
    }


ADDED_APPLE = (
    "def check_func(final_state):\n"
    "    return 'apple' in final_state['items']"
)
ALREADY_TRUE = (
    "def check_func(final_state):\n"
    "    return isinstance(final_state['items'], dict)"
)
UNCHANGED_START = (
    "def check_func(final_state):\n"
    "    return final_state['opened_at'] == initial_state['opened_at']"
)
BROKEN = "def check_func(final_state):\n    return final_state['missing']['key']"


def sources() -> dict[str, list[dict]]:
    worlds = [
        _world("env_1_rl", "Shop", SHOP_SOURCE, [
            _tool("add_item", name="string", qty="integer"),
            _tool("list_items"),
            _tool("explode"),
        ]),
        _world("env_2_rl", "Hostile", HOSTILE_SOURCE, [
            _tool("write_file", path="string"),
            _tool("read_file", path="string"),
            _tool("connect", host="string", port="integer"),
            _tool("spawn", command="string"),
            _tool("fork"),
            _tool("spin"),
            _tool("hog"),
            _tool("environment"),
        ]),
        _world("env_3_sft", "Shop", SHOP_SOURCE, [
            _tool("add_item", name="string", qty="integer"),
        ]),
    ]
    rl = [
        {
            "env_id": "env_1_rl",
            "task_id": "env_1_rl-task_1",
            "task": "Add an apple to the shop.",
            "init_config": {"items": {}},
            "checklist_with_func": [
                {"check_func": ADDED_APPLE, "check_item": "apple added"},
                {"check_func": ALREADY_TRUE, "check_item": "items is a mapping"},
                {"check_func": BROKEN, "check_item": "cannot run"},
            ],
        },
        {
            "env_id": "env_2_rl",
            "task_id": "env_2_rl-task_1",
            "task": "Take notes.",
            "init_config": {"notes": {}},
            "checklist_with_func": [
                {"check_func": (
                    "def check_func(final_state):\n"
                    "    return bool(final_state['notes'])"
                ), "check_item": "a note exists"},
            ],
        },
        {
            "env_id": "env_1_rl",
            "task_id": "env_1_rl-task_2",
            "task": "Keep the shop as it is.",
            "init_config": {"items": {"pear": {"qty": 1}}},
            "checklist_with_func": [
                {"check_func": UNCHANGED_START, "check_item": "opened_at unchanged"},
            ],
        },
    ]
    sft = [
        {
            "env_id": "env_3_sft",
            "env_class_name": "Shop",
            "task_id": "env_3_sft-task_1",
            "task": "Stock the shelves.",
            "init_config": {"items": {}},
        },
    ]
    return {"env_meta.json": worlds, "rl_scen.json": rl, "sft_scen.json": sft}


# Index 0: the shop (graded), 1: the hostile world (graded), 2: the shop with
# its one check already true (graded, unwinnable), 3: the ungraded shop.
SHOP, HOSTILE, SATISFIED, UNGRADED = 0, 1, 2, 3


def corpus() -> Corpus:
    return build_corpus(sources(), {
        "rl": [0, 1], "sft_graded": [2], "sft_ungraded": [0],
    })


def write_sources(directory: Path) -> None:
    for name, records in sources().items():
        (directory / name).write_text(json.dumps(records), encoding="utf-8")
