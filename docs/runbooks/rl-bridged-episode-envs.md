# RL episode envs served through the verifiers bridge

A signed-episode env of an RL order that a gateway serves through the sandbox's verifiers bridge needs no
code in this repository. Its order entry carries a public `catalogue`, and validators and miners build
the same task source from it (`reliquary.environment.bridged_episode`):

| catalogue field | what it is |
| --- | --- |
| `package` | `name==version`: the env distribution the gateways serve |
| `rl_split` | the served split the RL run draws from (the entry's `episode.split`) |
| `options` | the gateways' `episode_env_options[<sandbox env>]`, **exactly** as the gateways have it |
| `image_lock_sha256` | the digest of the package's `sandbox-images.lock.json` |

The catalogue is public, because it is part of the order. Never put a credential in `options`: the
contract refuses keys and values that look like one.

## 1. Describe the env from the installed wheel

Install the published wheel (`pip install --no-deps`), not a source tree. The identity that record 0
names is computed from the installed files. Then run:

```bash
python -m reliquary.environment.bridged_episode describe \
  --name tmax_v1 --package reliquary-terminal==0.3.0 --rl-split tmax_rl \
  --options tmax-options.json --episode tmax-episode.json --cooldown 10
```

`tmax-options.json` is the file the gateways are configured from:

```json
{"tools": ["bash"], "splits": {"tmax_rl": {"split": "tmax_rl"}}}
```

`tmax-episode.json` holds the run's choices for the episode block: `sandbox_env`, `tools`, `max_turns`,
`max_tokens_per_turn`, `max_episode_tokens` and `budgets`.

The command prints the entry and its catalogue:

- `version` is the manifest digest of the identity, options, lock, split and rows.
- `episode.env_package` is the bridged identity `name==version+g<hash>.vfb<N>`.
- `dataset.rows` is the RL split's size.

Review them, set `share_bps`, and put `{**environment, "catalogue": catalogue}` in the order.

## 2. Configure the gateways with the same options

The gateways must serve the env under the entry's `episode.sandbox_env`, with **the same options object**.
Generate both from the one options file above:

```bash
EPISODE_ENVS='{"reliquary-terminal": "verifiers:reliquary-terminal==0.3.0"}'
EPISODE_ENV_OPTIONS='{"reliquary-terminal": {"tools": ["bash"], "splits": {"tmax_rl": {"split": "tmax_rl"}}}}'
```

TMax's RL tasks are the `tmax_rl` split. A gateway that serves `{"tmax": {"split": "tmax"}}` publishes a
different options digest (`env_options_sha256` in its capacity report). The validator then places no session
on it:

- at start the env is degraded (`environment_unserved`, with an ALERT);
- its servable task set is empty.

Each gateway must also have the lock's images present. The TMax lock pins one public base image, by
digest:

```
ghcr.io/reliquadotai/reliquary-tmax-base@sha256:3d10148ef334ddea2ad5c7f1ea5d541fe8aa7877fe97a35bd6f706238e253f15
```

Pull images by digest: an image loaded from an archive has no registry digest and never counts as present.

## 3. What the validator checks at start

A catalogue env is accepted in `--environments` with the protocol's group count. It is neither installed in
the image nor in the protocol profile. The validator then:

1. builds the source from the catalogue;
2. refuses to start when any of these differs from the order:
   - the installed identity, compared with `episode.env_package`;
   - the installed lock, compared with `image_lock_sha256`;
   - the manifest, compared with the entry's `version`;
   - the row count, compared with `dataset.rows`.

A catalogue env whose name is also an installed env is refused. A bridged env needs a name of its own.

## 4. Mining it

Miners need no code for a catalogue env either. `reliquary mine-episodes --episode-envs <name>` accepts the
name of an env the order's catalogue describes. When the stack is built, the miner reads the announcement and
builds the same task source as the validator, from the same catalogue, over its own install of the env
package. It refuses to start (exit 4) when:

- the installed identity is not the order's `episode.env_package`;
- the installed lock is not the catalogue's `image_lock_sha256`;
- the announcement does not name the env.

So a miner installs exactly the pinned wheel (`pip install --no-deps`) and the reliquary-sandbox bridge. The
legacy `mine` command never mines these envs and refuses their names.

An env added to a running run (a schedule row carrying `added`) is mined the same way, from the row's
catalogue. The miner binds its prompts once, when its stack is built, so:

1. wait until the announcement names the new env;
2. install its pinned package;
3. add its name to `--episode-envs` and restart the miner.

One miner engine serves only envs whose episode caps (`max_episode_tokens`, `max_tokens_per_turn`) are equal.
An added env whose caps differ from a launch env's is refused beside it at start: mine it with a miner process
of its own (`--episode-envs <added env>` alone, on its own `--generate-port`).
