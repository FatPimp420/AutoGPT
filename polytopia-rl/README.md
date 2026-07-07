# polytopia-rl

Reinforcement learning on the [GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes)
(Polytopia-like) framework: an in-process Java↔Python bridge, a Gym-style
step-wise environment, and a self-play PPO training loop.

## Layout

```
Tribes/                          # upstream framework (unmodified clone)
java/src/core/game/RLGameRunner.java   # step-wise game driver (same package as GameState
                                 #   to reach its package-private turn lifecycle)
java/src/RLSmokeTest.java        # pure-Java full-game sanity check
polytopia_rl/bridge.py           # JPype JVM startup + TribesEnv wrapper
polytopia_rl/encoder.py          # int observations -> float planes / action features
polytopia_rl/model.py            # CNN state encoder + per-action scoring policy + value head
polytopia_rl/ppo.py              # PPO/GAE over ragged legal-action sets
train.py                         # self-play training entry point
```

## Setup

```bash
# 1. compile the framework and the shim
cd Tribes && javac -cp lib/json.jar -d out $(find src -name "*.java") && cd ..
javac -cp Tribes/out:Tribes/lib/json.jar -d java/out java/src/core/game/RLGameRunner.java

# 2. python env
python3 -m venv .venv
.venv/bin/pip install numpy jpype1
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Train

```bash
.venv/bin/python train.py --iterations 5 --rollout-steps 2048 --max-ticks 12 --eval-games 10
```

One policy plays both tribes (parameter-shared self-play). Rewards: terminal
win/loss (±1) plus a small dense score-difference shaping. `--max-ticks`
controls episode length (CAPITALS mode turn limit); raise it toward the
default 50 for real experiments. `--eval-games` pits the greedy policy
against the framework's built-in `RandomAgent`.

## Design notes

- **Step API**: `GameState.advance(action, computeActions)` is the framework's
  real forward model and handles full turn transitions. `RLGameRunner` adds
  what the whole-game `Game.run()` loop normally owns: tick increments when
  the turn rotation wraps, forced `EndTurn` when the state demands it, and
  optional in-process Java bot opponents (`random`/`simple`/`donothing`).
- **JVM working directory** must be the `Tribes/` repo root — the level
  generator loads `terrainProbs.json` and friends via relative paths.
  `bridge.start_jvm()` chdirs there before starting the JVM.
- **Action space**: legal actions are scored individually (pointer-network
  style) from per-action feature vectors (type one-hot, source/target
  coordinates, extra ordinal), so the variable-size legal set needs no fixed
  global action space; PPO minibatches pad ragged action sets and mask.
- **Observation**: per-tile one-hot planes for terrain/resource/building/unit
  plus ownership (player-relative), HP, city, road, and visibility channels,
  and a small global scalar vector (tick, stars, scores, counts).

## Watching it play

There's no display in the training container, so instead of the framework's
Swing GUI, record a game and view it in the browser:

```bash
.venv/bin/python record_game.py --out replay.json   # self-play with the checkpoint
# inject replay.json into viewer_template.html at /*__REPLAY__*/null and open it
```

The viewer shows the full board (terrain, units, cities, resources, roads),
the acting tribe's chosen action with the policy's value estimate, a
click-to-jump action log, and play/pause/scrub controls (space, ←/→).
