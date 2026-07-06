"""Encoders: raw integer observations from the bridge -> float tensors for the net.

Grid encoding is player-relative (own vs enemy planes), so one network can play
any seat. Action descriptors become fixed-size feature vectors scored per-action
by the policy head (pointer-style), which handles the variable-size legal-action
set without a fixed global action space.
"""

import numpy as np

HP_SCALE = 40.0
LEVEL_SCALE = 10.0
EXTRA_SCALE = 30.0
SCORE_SCALE = 10000.0


class Encoder:
    def __init__(self, type_sizes, board_size, max_ticks):
        self.sizes = type_sizes
        self.n = board_size
        self.max_ticks = max_ticks
        t, r, b, u = (type_sizes[k] for k in ("terrain", "resource", "building", "unit"))
        # terrain + resource + building + unit-type one-hots, then:
        # unit own/enemy, hp, maxhp, fresh, veteran, territory own/enemy,
        # city level, road, visible
        self.grid_channels = t + r + b + u + 11
        self.scalar_dim = 8
        self.action_dim = type_sizes["action"] + 9

    def encode_grid(self, grid, player):
        """grid: int32 (13, n, n) from the bridge; player: observing player id."""
        t, r, b, u = (self.sizes[k] for k in ("terrain", "resource", "building", "unit"))
        n = self.n
        out = np.zeros((self.grid_channels, n, n), dtype=np.float32)
        c = 0
        for size, ch in ((t, 0), (r, 1), (b, 2), (u, 3)):  # one-hot channels
            vals = grid[ch]
            mask = vals >= 0
            xs, ys = np.nonzero(mask)
            out[c + vals[mask], xs, ys] = 1.0
            c += size
        owner = grid[4]
        out[c][owner == player] = 1.0
        out[c + 1][(owner >= 0) & (owner != player)] = 1.0
        out[c + 2] = grid[5] / HP_SCALE
        out[c + 3] = grid[6] / HP_SCALE
        out[c + 4] = grid[7]
        out[c + 5] = grid[8]
        terr_owner = grid[9]
        out[c + 6][terr_owner == player] = 1.0
        out[c + 7][(terr_owner >= 0) & (terr_owner != player)] = 1.0
        out[c + 8] = grid[10] / LEVEL_SCALE
        out[c + 9] = grid[11]
        out[c + 10] = grid[12]
        return out

    def encode_scalars(self, scalars):
        """{tick, activeId, playerId, stars, ownScore, bestOppScore, nCities, nUnits, nTechs, size}"""
        tick, _, pid, stars, own, opp, ncities, nunits, ntechs, _ = (float(v) for v in scalars)
        return np.array([
            tick / self.max_ticks, stars / 50.0, own / SCORE_SCALE, opp / SCORE_SCALE,
            (own - opp) / SCORE_SCALE, ncities / 10.0, nunits / 20.0, ntechs / 25.0,
        ], dtype=np.float32)

    def encode_actions(self, actions):
        """actions: int32 (N, 8) descriptors -> float32 (N, action_dim)."""
        n_act = actions.shape[0]
        a_types = self.sizes["action"]
        out = np.zeros((n_act, self.action_dim), dtype=np.float32)
        out[np.arange(n_act), actions[:, 0]] = 1.0
        f = a_types
        coords = actions[:, 1:5].astype(np.float32)
        has = (coords >= 0).astype(np.float32)
        out[:, f:f + 4] = np.where(coords >= 0, coords / self.n, 0.0)
        out[:, f + 4] = has[:, 0]          # has source pos
        out[:, f + 5] = has[:, 2]          # has target pos
        extra = actions[:, 5].astype(np.float32)
        out[:, f + 6] = np.where(extra >= 0, extra / EXTRA_SCALE, 0.0)
        out[:, f + 7] = (extra >= 0).astype(np.float32)
        # Manhattan distance source->target where both exist
        both = (has[:, 0] > 0) & (has[:, 2] > 0)
        dist = np.abs(coords[:, 0] - coords[:, 2]) + np.abs(coords[:, 1] - coords[:, 3])
        out[:, f + 8] = np.where(both, dist / (2.0 * self.n), 0.0)
        return out

    def encode(self, obs):
        return (
            self.encode_grid(obs["grid"], obs["player"]),
            self.encode_scalars(obs["scalars"]),
            self.encode_actions(obs["actions"]),
        )
