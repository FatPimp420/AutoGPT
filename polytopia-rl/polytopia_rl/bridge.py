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

    # CONQUEST = SCORE's last-tribe-standing rule with the turn limit lifted.
    # A generous safety cap still applies so a stalemate can't run forever.
    CONQUEST_CAP = 300

    def __init__(self, tribes=("XIN_XI", "IMPERIUS"), game_mode="CAPITALS",
                 bot_types=None, max_ticks=None, map_size=0):
        cls = start_jvm()
        self.runner = cls()
        if map_size:  # 0 = Polytopia's default size for the player count
            self.runner.setMapSize(int(map_size))
        self.tribes = list(tribes)
        self.conquest = game_mode.upper() == "CONQUEST"
        # The Java runner treats any non-CAPITALS mode as SCORE (which ends when
        # only one tribe is un-eliminated) — exactly the Conquest rule.
        self.game_mode = "CAPITALS" if game_mode.upper() == "CAPITALS" else "SCORE"
        self.bot_types = list(bot_types) if bot_types else [""] * len(tribes)
        C = jpype.JClass("core.Constants")
        if self.conquest:
            C.MAX_TURNS = int(max_ticks) if max_ticks else self.CONQUEST_CAP
        elif self.game_mode == "SCORE":
            C.MAX_TURNS = int(max_ticks) if max_ticks else 30
        elif max_ticks is not None:  # CAPITALS
            C.MAX_TURNS_CAPITALS = int(max_ticks)
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

    # Progression stats (see RLGameRunner.getTickStats / getPlayerStats).
    def tick_stats(self):
        """Cheap per-turn aggregates: (total_cities, alive_tribes, total_stars)."""
        return tuple(int(v) for v in self.runner.getTickStats())

    PSTAT = ("stars", "score", "cities", "units", "techs", "kills",
             "tiles", "capital", "alive")

    def player_stats(self, pid):
        """End-of-game per-player snapshot as a dict (see PSTAT for keys)."""
        vals = [int(v) for v in self.runner.getPlayerStats(int(pid))]
        return dict(zip(self.PSTAT, vals))

    def player_counts(self):
        """Cheap per-player [cities, kills, alive] rows for reward shaping."""
        flat = [int(v) for v in self.runner.getPlayerCounts()]
        return [flat[i:i + 3] for i in range(0, len(flat), 3)]

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
