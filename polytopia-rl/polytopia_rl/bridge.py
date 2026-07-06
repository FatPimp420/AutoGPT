"""JPype bridge to the Tribes Java framework.

Starts an in-process JVM with the Tribes classes plus the RLGameRunner shim and
exposes a step-wise, multi-agent environment. The JVM working directory must be
the Tribes repo root (LevelGenerator reads terrainProbs.json etc. relatively),
so start_jvm() chdirs there before launching.
"""

import os
from pathlib import Path

import jpype
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRIBES_DIR = ROOT / "Tribes"
CLASSPATH = [str(TRIBES_DIR / "out"), str(TRIBES_DIR / "lib" / "json.jar"), str(ROOT / "java" / "out")]

_runner_cls = None


def start_jvm():
    """Starts the JVM (idempotent) and returns the RLGameRunner class."""
    global _runner_cls
    if _runner_cls is not None:
        return _runner_cls
    os.chdir(TRIBES_DIR)  # relative config paths (terrainProbs.json) resolve from here
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=CLASSPATH, convertStrings=False)
    _runner_cls = jpype.JClass("core.game.RLGameRunner")
    return _runner_cls


class TribesEnv:
    """Step-wise multi-agent wrapper around core.game.RLGameRunner.

    Players whose bot_types entry is "" are externally controlled; whichever of
    them is active is reported in the observation ("player"), and step() acts
    for that player. Rewards are computed by the caller (train loop) from
    scores/win status so different shaping schemes stay out of the env.
    """

    OBS_CHANNELS = 13
    ACTION_FIELDS = 8
    SCALAR_FIELDS = 10

    def __init__(self, tribes=("XIN_XI", "IMPERIUS"), game_mode="CAPITALS",
                 bot_types=None, max_ticks=None):
        cls = start_jvm()
        self.runner = cls()
        self.tribes = list(tribes)
        self.game_mode = game_mode
        self.bot_types = list(bot_types) if bot_types else [""] * len(tribes)
        if max_ticks is not None and game_mode.upper() == "CAPITALS":
            jpype.JClass("core.Constants").MAX_TURNS_CAPITALS = int(max_ticks)
        self.n_players = len(tribes)
        self.board_size = None

    # enum cardinalities for the encoder
    @property
    def type_sizes(self):
        cls = start_jvm()
        return {
            "terrain": int(cls.numTerrainTypes()),
            "resource": int(cls.numResourceTypes()),
            "building": int(cls.numBuildingTypes()),
            "unit": int(cls.numUnitTypes()),
            "action": int(cls.numActionTypes()),
        }

    def reset(self, level_seed, game_seed):
        self.runner.reset(int(level_seed), int(game_seed),
                          self.tribes, self.game_mode, self.bot_types)
        self.board_size = int(self.runner.getBoardSize())
        return self._observe()

    def step(self, action_idx):
        """Applies the action for the currently active external player."""
        self.runner.step(int(action_idx))
        return self._observe()

    def _observe(self):
        r = self.runner
        done = bool(r.isGameOver())
        if done:
            return {
                "done": True,
                "tick": int(r.getTick()),
                "scores": [int(r.getScore(i)) for i in range(self.n_players)],
                "win": [int(r.getWinStatus(i)) for i in range(self.n_players)],
            }
        pid = int(r.getActiveTribeID())
        n = self.board_size
        grid = np.asarray(r.getObservation(pid), dtype=np.int32).reshape(self.OBS_CHANNELS, n, n)
        scalars = np.asarray(r.getScalars(pid), dtype=np.int32)
        acts = np.asarray(r.getLegalActionData(), dtype=np.int32).reshape(-1, self.ACTION_FIELDS)
        return {
            "done": False,
            "player": pid,
            "tick": int(r.getTick()),
            "grid": grid,
            "scalars": scalars,
            "actions": acts,
            "scores": [int(r.getScore(i)) for i in range(self.n_players)],
        }
