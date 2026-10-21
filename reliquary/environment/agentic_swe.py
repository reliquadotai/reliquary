"""What an agentic SWE corpus job pins, and the prompt source behind it.

The bash harness itself (system prompt, tool specs, call filter, network notice, the
Qwen3.8 turn renderer) is `reliquary_harness`'s `BASH_VF_B2E4E81` spec, the one
definition miners and validators import; the names below are that spec's, re-exported
for this module's callers. `renderers`, `verifiers` and `reliquary_swe` are optional:
imported inside functions only.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import json
import subprocess
import threading
from collections.abc import Sequence

from reliquary_harness import qwen as _harness_qwen
from reliquary_harness.spec import (
    BASH_SURFACE_V1, CLAUDE_CODE_V1, CODEX_V1, MINI_SWE_V2, OPENHANDS_XML_V1, VERIFIERS_COMMIT, get_spec,
)
from reliquary_harness.spec import BASH_VF_B2E4E81 as HARNESS
from reliquary_harness.variant import variant_count, variant_index

from reliquary.environment.agentic.types import EpisodeTask

HARNESS_DIGEST = "9a6b27dd764cffee512f77f430747c40c220fd6492b30696b8c89e2fb20f9fca"
"""The bash harness this build renders, checks and mines with (`HARNESS.digest`): an
installed reliquary-harness with another digest is not this build's harness."""
HARNESS_DIGESTS = {
    HARNESS.key: HARNESS_DIGEST,
    BASH_SURFACE_V1.key: "4943b1efec1c88691a727f819538b00ef321e6eb8c274e79d2d1ddca3306c4be",
    MINI_SWE_V2.key: "cecffb2631c01e2dd415a4e9cb76d79717d1a0b67bd0187422a658ef9173f15d",
    OPENHANDS_XML_V1.key: "0b1a9b92ccb3b43e8954a15afa5ab6c5f14e57190a734118f817ce7da848ff4d",
    CLAUDE_CODE_V1.key: "2763bcd5c36ac027e9d5e24cad80c9c606fbbb26e07bcb2f192ecb9ba464e40d",
    CODEX_V1.key: "00d2979c6a783bb71ffb0c553881155cb61f8868ea20d61f7cc0cd75fd99a61b",
}
"""Every harness spec this build serves, by `id@version`, at the digest it was built against.
A job or an RL episode env names one with `{id, version, digest}`; one naming none is
`bash-vf-b2e4e81@1`."""
SUPPORTED_RENDERER = HARNESS.renderer
RENDERERS_VERSION = SUPPORTED_RENDERER.rsplit("@", 1)[1]
SUPPORTED_VERIFIERS = VERIFIERS_COMMIT

BASH_SENTENCE = HARNESS.system_prompt["fragments"]["bash"]
EDIT_SENTENCE = HARNESS.system_prompt["fragments"]["edit"]
BASH_SYSTEM_PROMPT = HARNESS.system_prompt_for(("bash", "edit"))
BASH_HARNESS_TOOLS: tuple[dict, ...] = HARNESS.tools
# The restricted-network notice (verifiers' dialects/base.py CAPABILITY_NOTICE): its
# interception appends it to the first user message of every request whose runtime
# restricts egress, which the replay episode config always does (`block: ["*"]`). The
# validator's render ALWAYS uses the spec's text (ruling P13); the miner refuses to mine
# when its verifiers differs (`agentic_episode.network_notice_refusal`).
PINNED_NETWORK_NOTICE = HARNESS.network_notice["text"]
pinned_tool_calls = _harness_qwen.pinned_tool_calls


def harness_tools(tools) -> tuple[dict, ...]:
    """The tool specs the bash harness sends for `tools` (bash always, edit optionally):
    the renderer must parse with exactly these, since parameter coercion follows their
    schemas. Order and repeats do not matter; an unknown name, or edit without bash, is
    refused."""
    return HARNESS.tool_specs(tools)


def harness_system_prompt(tools) -> str:
    return HARNESS.system_prompt_for(tools)


def with_network_notice(content: str) -> str:
    """`append_user_notice` on a string user message."""
    return HARNESS.with_network_notice(content)


def harness_refusal() -> str | None:
    """Why the installed reliquary-harness is not this build's harness, or None."""
    for key, pinned in HARNESS_DIGESTS.items():
        installed = get_spec(*key.split("@", 1)).digest
        if installed != pinned:
            return (f"reliquary-harness {key} is installed with digest {installed}, "
                    f"this build needs {pinned}")
    return None


def harness_pin_refusal(pin, *, variants: bool) -> str | None:
    """Why this build cannot serve a contract naming harness `pin` (`{id, version, digest}`,
    or None for the compatibility harness), or None. Any harness this build serves plays on
    any task (corpus, eval, RL window or lease). `variants`: whether the caller draws a
    surface variant per episode from a validator seed (an RL lease does; a corpus job or an
    RL window has none, so a harness with variants is refused there). A refusal to serve,
    never a judgment."""
    refusal = harness_refusal()
    if refusal or pin is None:
        return refusal
    key = f"{pin['id']}@{pin['version']}"
    if key not in HARNESS_DIGESTS:
        return f"harness {key} is not one this build serves ({sorted(HARNESS_DIGESTS)})"
    if pin["digest"] != HARNESS_DIGESTS[key]:
        return f"the contract pins harness {key} at {pin['digest']}, this build serves {HARNESS_DIGESTS[key]}"
    if not variants and variant_count(get_spec(pin["id"], pin["version"])) != 1:
        return f"harness {key} draws surface variants per episode: an RL lease's, not a corpus job's"
    return None


def harness_of(pin):
    """The spec a contract names (`{id, version, digest}`; None: the compatibility harness).
    Callers checked it first (`harness_pin_refusal`)."""
    return HARNESS if pin is None else get_spec(pin["id"], pin["version"])


def job_harness(job):
    """The spec a corpus or eval job's episodes play (`episode.harness_spec`; none: the
    compatibility harness)."""
    return harness_of(getattr(getattr(job, "episode", None), "harness_spec", None))


def job_turn_renderer(job, checkpoint_dir: str) -> QwenTurnRenderer:
    """The turn renderer of a signed job: its tools, in its harness. The miner, the intake
    and the export all build this one."""
    return load_turn_renderer(checkpoint_dir, tools=job.episode.sandbox.tools, spec=job_harness(job))


def policy_harness_pin(policy):
    """An episode env policy's harness pin (`EpisodeEnvPolicy.harness_pin`), None when it names none."""
    pin = getattr(policy, "harness_pin", None)
    return pin() if callable(pin) else None


def lease_variant(spec, lease) -> int:
    """The surface variant every episode of an RL lease plays: drawn from the lease's draw
    nonce (the validator's secret-derived, the miner cannot choose it), the lease and its
    prompt, so the 16 rollouts of a group share it and both sides compute it alone. A spec
    without axes has variant 0 only."""
    if variant_count(spec) == 1:
        return 0
    return variant_index(spec, seed=str(lease.nonce), engagement=f"rl-lease:{lease.lease_id}",
                         task_index=int(lease.prompt_idx))


def policy_harness_choices(policy) -> tuple:
    """((pin, weight), ...) an episode env policy plays: a weighted mix in its block's order, else its one
    harness (None: the compatibility harness) at weight 1."""
    choices = getattr(policy, "harness_choices", None)
    return tuple(choices()) if callable(choices) else ((policy_harness_pin(policy), 1),)


def policy_default_harness(policy):
    """The spec of a policy's first harness: its only one, or its mix's first (the renderer a check builds
    before any lease; every harness of one tokenizer stops on the same ids)."""
    return harness_of(policy_harness_choices(policy)[0][0])


def harness_mix_index(choices, lease) -> int:
    """Which harness of a weighted mix an RL lease plays: `H(nonce ‖ engagement ‖ prompt ‖ mix) mod Σ
    weights`, each part length-prefixed, the mix being its (key, digest, weight) list. Drawn from the lease's
    draw nonce (the validator's secret-derived), so the miner cannot choose it, both sides compute it alone,
    and the 16 rollouts of a group share it. One harness: 0."""
    import hashlib

    if len(choices) == 1:
        return 0
    mix = json.dumps([[harness_of(pin).key, harness_of(pin).digest, int(weight)] for pin, weight in choices],
                     separators=(",", ":")).encode()
    digest = hashlib.sha256()
    for part in (str(lease.nonce).encode(), f"rl-lease:{lease.lease_id}".encode(),
                 str(int(lease.prompt_idx)).encode(), b"harness-mix", mix):
        digest.update(len(part).to_bytes(8, "big") + part)
    point = int.from_bytes(digest.digest(), "big") % sum(int(weight) for _, weight in choices)
    for index, (_, weight) in enumerate(choices):
        if point < int(weight):
            return index
        point -= int(weight)
    raise AssertionError("unreachable: the point is below the weights' sum")


def lease_surface(policy, lease):
    """(spec, variant) every episode of an RL lease plays: the mix's harness drawn for the lease (its only
    one without a mix), then that harness's surface variant (`lease_variant`)."""
    choices = policy_harness_choices(policy)
    spec = harness_of(choices[harness_mix_index(choices, lease)][0])
    return spec, lease_variant(spec, lease)


def harness_value_refusal(value) -> str | None:
    """Why this build cannot serve an RL episode block's `harness`: one pin (None: the compatibility
    harness), or a weighted mix whose every harness it must serve; or None."""
    if not isinstance(value, list):
        return harness_pin_refusal(value, variants=True)
    for entry in value:
        refusal = harness_pin_refusal({k: v for k, v in entry.items() if k != "weight"}, variants=True)
        if refusal:
            return refusal
    return None


def policy_harness_refusal(policy) -> str | None:
    """`harness_value_refusal` of a typed policy (every harness it may draw)."""
    for pin, _ in policy_harness_choices(policy):
        refusal = harness_pin_refusal(pin, variants=True)
        if refusal:
            return refusal
    return None


def _dist_commit(name: str) -> str | None:
    """The commit a distribution was installed from: its VCS pin, or the HEAD
    of the checkout an editable install points at. None when unknowable."""
    try:
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    info = json.loads(raw)
    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return commit
    url = info.get("url") or ""
    if url.startswith("file://"):
        checkout = url[len("file://"):]
        git = ["git", "-c", "safe.directory=*", "-C", checkout]
        try:
            head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=10)
            dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True,
                                   timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return None
        # An editable checkout with local changes is not the pinned code,
        # whatever its HEAD says: report nothing rather than the commit.
        if head.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip():
            return head.stdout.strip()
    return None


def installed_env_commit() -> str | None:
    return _dist_commit("reliquary-swe")


def installed_verifiers_commit() -> str | None:
    return _dist_commit("verifiers")


def _renderers_version() -> str | None:
    try:
        return importlib.metadata.version("renderers")
    except importlib.metadata.PackageNotFoundError:
        return None


def episode_support_refusal(episode, *, need_verifiers: bool) -> str | None:
    """Why this binary cannot serve `episode`, or None. ``need_verifiers`` for
    the processes that run episodes (miner, grade executor)."""
    pin = getattr(episode, "harness_spec", None)
    refusal = harness_pin_refusal(pin, variants=False)
    if refusal:
        return refusal
    if getattr(episode, "sandbox", None) is None and harness_of(pin) is not HARNESS:
        # The legacy Docker replay runs verifiers' own bash harness (step 7 retires it).
        return f"a replay (non signed_sandbox) job plays {HARNESS.key} only, not {harness_of(pin).key}"
    if episode.renderer != SUPPORTED_RENDERER:
        return f"episode.renderer {episode.renderer!r} is not {SUPPORTED_RENDERER!r}"
    if episode.verifiers != SUPPORTED_VERIFIERS:
        return f"episode.verifiers {episode.verifiers} is not the supported {SUPPORTED_VERIFIERS}"
    if _renderers_version() != RENDERERS_VERSION:
        return f"renderers {RENDERERS_VERSION} is not installed (found {_renderers_version()})"
    installed = installed_env_commit()
    if installed != episode.env.version:
        return (f"reliquary-swe is installed at {installed}, the job pins "
                f"{episode.env.version}")
    if need_verifiers and installed_verifiers_commit() != SUPPORTED_VERIFIERS:
        return (f"verifiers is installed at {installed_verifiers_commit()}, "
                f"the job pins {SUPPORTED_VERIFIERS}")
    return None


class SweSource:
    """SWE-smith tasks by source index: the instance id and the user prompt,
    nothing else (a full task set costs over 5 GB of memory as tasks)."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[tuple[str, str]]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def instance_id(self, index: int) -> str:
        return self._rows[index][0]

    def prompt(self, index: int) -> str:
        """The first user message as the model sees it: the task prompt plus
        verifiers' restricted-network notice (ruling P13)."""
        return with_network_notice(self._rows[index][1])

    def task_for(self, index: int) -> EpisodeTask:
        instance_id, prompt = self._rows[index]
        return EpisodeTask(id=instance_id, prompt=prompt, tools=())


@functools.lru_cache(maxsize=4)
def load_swe_source(num_images: int) -> SweSource:
    """The task set `reliquary-swe` builds for `num_images` (split train), in
    its own deterministic order: index i here is task i on every machine."""
    from reliquary_swe import corpus
    from reliquary_swe.taskset import PROMPT

    rows = corpus.load_swesmith_rows(num_images)
    return SweSource([(row.instance_id,
                       PROMPT.format(workdir=row.workdir, problem_statement=row.problem_statement))
                      for row in rows])


class SignedSweSource:
    """The prompts of a signed-sandbox job: the bridged task's prompt
    (`bridged_swe.prompt_of(split, index)`) exactly, with no restricted-network notice
    (the sandbox bridge runs the harness in a subprocess runtime, which adds none).

    `rows` is the job's `RowMap` (`reliquary.eval.rows.job_row_map`). An eval job's
    references map row `i` to `(rows.split, rows.index(i))`, and are checked to rebuild
    into the instances they name on first use. The identity reads row `i` of its split
    as itself, any row of it (a training job's SWE-smith rows, by index, as always: the
    job's own bounds are its ownership checks', not the source's)."""

    def __init__(self, rows, *, prompt_of=None, row_of=None) -> None:
        self.split, self.rows = rows.split, rows
        self._prompt_of = prompt_of
        self._row_of = row_of
        self._checked = rows.is_identity
        self._check_lock = threading.Lock()
        self._prompts = functools.lru_cache(maxsize=4096)(self._prompt)
        self._instance_ids = functools.lru_cache(maxsize=4096)(self._instance_id)

    def __len__(self) -> int:
        """The job's rows with references; the split's rows the bridge serves for the
        identity."""
        if not self.rows.is_identity:
            return len(self.rows)
        from reliquary.environment.bridged_swe import split_length

        return split_length(self.split)

    def __bool__(self) -> bool:
        return True                                   # `source or default`: never a length read

    def check(self) -> None:
        """That every row's reference rebuilds into the instance it names (ValueError
        naming the first that does not); once, and nothing to check for the identity."""
        if self._checked:
            return
        with self._check_lock:
            if not self._checked:
                self.rows.check_instances(self._source_instance_id)
                self._checked = True

    def task_ref(self, index: int):
        """The task row `index` runs (a `reliquary.eval.rows.TaskRef`): the session
        token's `(split, index)`."""
        from reliquary.eval.rows import TaskRef

        if self.rows.is_identity:
            return TaskRef(self.split, int(index), None)
        self.check()
        return self.rows.ref(int(index))

    def _index(self, index: int) -> int:
        return self.task_ref(index).index

    def _prompt(self, index: int) -> str:
        prompt_of = self._prompt_of
        if prompt_of is None:
            from reliquary.environment.bridged_swe import prompt_of
        return prompt_of(self.split, self._index(index))

    def prompt(self, index: int) -> str:
        return self._prompts(int(index))

    def instance_id(self, index: int) -> str:
        return self._instance_ids(int(index))

    def _instance_id(self, index: int) -> str:
        return self._source_instance_id(self.split, self._index(index))

    def _source_instance_id(self, split: str, index: int) -> str:
        row_of = self._row_of
        if row_of is None:
            from reliquary.environment.bridged_swe import row_of
        return row_of(split, int(index))[1].instance_id

    def task_for(self, index: int) -> EpisodeTask:
        return EpisodeTask(id=self.instance_id(index), prompt=self.prompt(index), tools=())


def job_signed_source(job, *, prompt_of=None, row_of=None) -> SignedSweSource:
    """The one source of a signed-sandbox job, for its intake, grader, export and miner.

    An eval job (its prompt source names a frozen set of row references) maps row `i`
    to the set's `i`-th reference at the set's split, and is checked here against the
    bridged env (ValueError on a stale set). Any other job keeps its split's rows by
    index, as before. Build it once per job and process: the check reads every row."""
    from reliquary.eval.rows import job_row_map

    source = SignedSweSource(job_row_map(job), prompt_of=prompt_of, row_of=row_of)
    source.check()
    return source


def installed_env_package(package: str) -> str:
    """The `env_package` a gateway serving this install writes into record 0:
    `<distribution>==<version>+g<16 hex>.vfb<bridge version>`, the bridged identity
    (`bridged_env_package_of`), computed by reliquary-sandbox's own task registry over
    the taskset's installed files (one source of truth)."""
    from reliquary.environment.bridged_swe import installed_env_package as bridged

    return bridged(package)


def sandbox_support_refusal(episode, *, need_bridge: bool = False) -> str | None:
    """Why this process cannot serve a signed-sandbox `episode`, or None:
    SWE-smith at the 20 images the bridge serves (a `train` job; an eval job's split is
    served at its own count), reliquary-sandbox at the job's commit,
    the env package at the exact version and code digest record 0 will carry, and the
    verifiers bridge (always: every process reads the task through it; `need_bridge` is
    kept for callers), which must load the installed package (an editable install is
    refused here, as the gateway refuses it)."""
    from reliquary import sandbox as sandbox_support

    if getattr(episode, "sandbox", None) is None:
        return "the episode is not a signed_sandbox one"
    refusal = harness_pin_refusal(getattr(episode, "harness_spec", None), variants=False)
    if refusal:
        return refusal
    try:
        sandbox_support.require_sandbox()
    except sandbox_support.SandboxUnavailable as exc:
        return str(exc)
    from reliquary.corpus.job import BRIDGED_SWESMITH_IMAGES

    if (getattr(episode.env, "split", "train") == "train"
            and getattr(episode.env, "num_images", BRIDGED_SWESMITH_IMAGES)
            != BRIDGED_SWESMITH_IMAGES):
        return (f"a signed reliquary-swe job serves SWE-smith at its default "
                f"{BRIDGED_SWESMITH_IMAGES} images (the split the bridge serves)")
    refusal = sandbox_support.sandbox_commit_refusal(episode.sandbox.sandbox_commit)
    if refusal:
        return refusal
    package = episode.env.package
    pinned = episode.sandbox.env_package
    try:
        installed = f"{package}=={importlib.metadata.version(package)}"
    except importlib.metadata.PackageNotFoundError:
        return f"{package} is not installed"
    if installed != pinned.rsplit("+g", 1)[0]:
        return f"{installed} is installed, the job pins {pinned}"
    try:
        importlib.import_module(package.replace('-', '_'))
    except Exception as exc:  # an import-time pin check raises RuntimeError, not ImportError
        return (f"{package.replace('-', '_')} cannot be imported ({type(exc).__name__}: {exc}): "
                f"install {package} at the job's commit")
    try:
        identity = installed_env_package(package)
    except Exception as exc:  # an unreadable install cannot be the pinned code
        return f"cannot compute the installed {package} identity ({exc})"
    if identity != pinned:
        return f"{identity} is installed, the job pins {pinned}"
    try:
        importlib.import_module("reliquary_sandbox_verifiers")
    except Exception as exc:  # the bridge refuses an unpinned verifiers with RuntimeError
        return f"the verifiers bridge cannot load: {exc}"
    try:
        from reliquary.environment import bridged_swe

        bridged_swe.bridged_env()
    except Exception as exc:  # e.g. an editable install: the bridge serves packaged tasksets only
        return f"the bridge cannot serve the installed {package} ({type(exc).__name__}: {exc})"
    return None


def _refused_span(exc: BaseException) -> BaseException:
    """A span the pinned parser raised on is refused as `bad_turns`."""
    from reliquary.corpus.checks import REASON_BAD_TURNS
    from reliquary.corpus.trajectory_parse import TrajectoryRefused

    if isinstance(exc, TrajectoryRefused):
        return exc
    return TrajectoryRefused(REASON_BAD_TURNS, {
        "why": "the pinned parser failed on a span", "error": type(exc).__name__})


class QwenTurnRenderer(_harness_qwen.QwenTurnRenderer):
    """`trajectory_parse.TurnRenderer` over the pinned `renderers` Qwen3.8
    renderer and a bash harness spec (default the compatibility one) in one of its surface
    variants: reliquary-harness's renderer, refusing a span the parser raises on with
    `TrajectoryRefused` (`bad_turns`)."""

    def __init__(self, renderer, tools=("bash", "edit"), *, spec=HARNESS, variant: int = 0) -> None:
        super().__init__(renderer, tools, spec=spec, variant=variant, refuse=_refused_span)


def load_turn_renderer(checkpoint_dir: str, tools=("bash", "edit"), *, spec=None,
                       variant: int = 0) -> QwenTurnRenderer:
    """The renderer the train client builds for this checkpoint
    (`create_renderer(load_tokenizer(model), Qwen38RendererConfig())`), over the
    job's tools, in `spec`'s surface `variant` (default: the compatibility harness)."""
    spec = spec or HARNESS
    return QwenTurnRenderer(_harness_qwen.build_renderer(checkpoint_dir, spec), tools=tools, spec=spec,
                            variant=variant)
