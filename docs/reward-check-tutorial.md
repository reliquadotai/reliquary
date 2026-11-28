# Test a verifiable reward before LLM reinforcement learning

Before spending compute on reinforcement learning, check whether your reward
distinguishes the behavior you want from plausible mistakes. A grader can be
deterministic and still reward the wrong answer.

This Reliquary example tests a reward offline. The fixtures are hand-written
examples, not measured model outputs or a benchmark. Running it requires no model,
GPU, network connection or Reliquary account.

## Start with a task you can inspect

The task is to remove duplicate integers while preserving their first occurrence.
For input `[3, 1, 3]`, the correct answer is `[3, 1]`.

The response must be a JSON array of integers. Sorting the values into `[1, 3]` is
wrong, as is dropping the last distinct value or leaving duplicates. A boolean or
a floating-point value is not accepted as an integer.

Under this example's contract, a correct answer earns 1. A wrong answer or invalid
answer JSON earns 0. Invalid task inputs and checker errors raise an error rather
than becoming an ordinary model failure. A candidate response passed to the
checker must be text.

These boundaries matter: the reward must describe the intended task, and the
evaluation must distinguish a wrong answer from a broken task runner.

## Run the example

The [standalone program](../scripts/rl_reward_check_v1.py) requires Python 3.10 or
later and uses only the Python standard library. From the repository root, run:

```bash
python3 scripts/rl_reward_check_v1.py
```

No package installation is needed. The program prints a JSON report with the
result of each check and three example groups. A successful run reports
`"checks_passed": 8` and `"checks_total": 8`, and exits with status 0. Any check
whose observed reward differs from its expected reward makes the program exit
with status 1. Checker errors stop execution with an error.

The eight supplied checks cover correct output, order, duplicates, omitted values,
empty inputs, invented values, boolean values and invalid JSON. They are a small
starting set, not an exhaustive validation of a reward.

## Inspect groups as well as individual answers

A correct reward function does not ensure that every sampled group contains a
useful relative signal. The example evaluates three groups of four hand-written
responses to the same task:

| Example group | Rewards | Population variance | Relative reward contrast |
| --- | --- | --- | --- |
| All fail | `[0, 0, 0, 0]` | 0 | No |
| Mixed | `[1, 0, 1, 0]` | 0.25 | Yes |
| All pass | `[1, 1, 1, 1]` | 0 | No |

When rewards within a group are identical, there is no within-group reward
contrast for a group-relative objective. This observation does not imply that
every training term is zero or that a particular dataset guarantees improvement.
The groups illustrate one binary-reward task; they do not prescribe a universal
success-rate threshold.

The program reports population variance for these fixtures. It does not reproduce
a trainer's reward normalization, full objective or logged statistics. For the
training context, TRL documents [GRPO reward statistics and zero-standard-deviation
groups](https://huggingface.co/docs/trl/grpo_trainer#logged-metrics).

## Extend the checks for your workload

The program exposes `reward(task, response)`, a binary grader, and
`run_checks(grader=reward)`, which returns the report. Its `main(grader=reward)`
function also prints the report and returns the appropriate exit status.

Add cases to `CHECKS` as `(name, task, response, expected_reward)` tuples. Choose
the expected reward independently of the grader being tested. Include known
correct responses, plausible wrong responses, malformed outputs and edge cases
from your actual task. Include both positive and negative cases: a grader that
accepts every answer or rejects every answer must fail the checks.

Use `GROUPS` to inspect the reward contrast across candidate responses. These
fixtures are authored; use sampled model responses in a separate experiment to
measure what your starting checkpoint actually does.

Pin the grader version and record environment failures separately. Before tuning,
reserve a final evaluation set and compare checkpoints with the same questions,
settings and grading rules.

The example is a task-specific checker, not a training API integration or a secure
sandbox for executing generated code. A real workload needs its own runtime,
isolation, limits and evaluation design. Passing these checks establishes neither
that all valid solutions are recognized nor that RL will improve a model.

## Continue with Reliquary

The [RL preparation guide](https://www.reliqua.ai/research/prepare-tasks-for-llm-reinforcement-learning)
provides the wider workflow. The [checkpoint comparison guide](https://app.reliqua.ai/docs/guides/compare-llm-checkpoints)
includes an offline example of score and coverage checks. For project enquiries,
[Reliquary's RL service page](https://www.reliqua.ai/services/training) describes
the planned customer offering and waitlist. Public research runs are separate;
this example does not demonstrate customer RL execution.

The example and documentation are available under the repository's
[MIT license](../LICENSE.md).
