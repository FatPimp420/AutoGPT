"""Plays one requested exhibition game and writes a replay JSON.

Run as a subprocess (own JVM) so per-game settings like the turn limit never
leak into the training process.

Request JSON:
{
  "id": "g-1712345",
  "tribes": ["XIN_XI", "BARDUR", "OUMAJI"],       // 2-4 seats
  "seats":  ["policy", "simple", "random"],        // who controls each seat:
             // "policy" = trained net (sampled), "policy-greedy",
             // "simple" | "random" | "donothing" = built-in Java bots
  "mode": "CAPITALS",                              // or "SCORE"
  "max_ticks": 20,
  "seed": 12345                                    // optional
}

Usage: game_player.py --request req.json --out replay.json [--ckpt policy.pt]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from polytopia_rl.bridge import TribesEnv, start_jvm
from polytopia_rl.encoder import Encoder
from polytopia_rl.model import PolicyValueNet
from record_game import enum_names, action_text, snapshot

ROOT = Path(__file__).resolve().parent
BOT_SEATS = {"simple", "random", "donothing", "osla", "mcts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "policy.pt"))
    args = ap.parse_args()

    req = json.loads(Path(args.request).read_text())
    tribes = req["tribes"]
    seats = req["seats"]
    assert len(tribes) == len(seats) and 2 <= len(tribes) <= 4
    mode = req.get("mode", "CAPITALS")
    conquest = mode.upper() == "CONQUEST"
    max_ticks = int(req.get("max_ticks", 0)) or (TribesEnv.CONQUEST_CAP if conquest else 12)
    seed = int(req.get("seed") or time.time())
    rng = np.random.default_rng(seed)

    # The engine's default size table only covers up to 8 players; for larger
    # games we must supply a size or level generation would go out of bounds.
    map_size = int(req.get("map_size", 0))
    if map_size == 0 and len(tribes) > 8:
        map_size = 24 + 2 * (len(tribes) - 8)  # extend the real progression

    bot_types = [s if s in BOT_SEATS else "" for s in seats]
    env = TribesEnv(tribes=tribes, game_mode=mode, bot_types=bot_types,
                    max_ticks=max_ticks, map_size=map_size)
    env.runner.setAutoPlayBots(False)
    obs = env.reset(rng.integers(1 << 30), rng.integers(1 << 30))
    # tick-normalization horizon: real cap for CAPITALS/CONQUEST, else SCORE's 30
    enc_horizon = max_ticks if mode.upper() in ("CAPITALS", "CONQUEST") else 30
    encoder = Encoder(env.type_sizes, env.board_size, enc_horizon)

    net = None
    if any(s.startswith("policy") for s in seats):
        ckpt = torch.load(args.ckpt, weights_only=True)
        cfg = dict(ckpt["config"])
        cfg["board_size"] = env.board_size  # conv+adaptive pool is size-agnostic
        net = PolicyValueNet(**cfg)
        net.load_state_dict(ckpt["model"])
        net.eval()

    start_jvm()
    names = {k: enum_names(v) for k, v in {
        "action": "core.Types$ACTION", "terrain": "core.Types$TERRAIN",
        "resource": "core.Types$RESOURCE", "building": "core.Types$BUILDING",
        "unit": "core.Types$UNIT", "tech": "core.Types$TECHNOLOGY"}.items()}

    frames = [{"tick": obs["tick"], "player": obs["player"], "scores": obs["scores"],
               "action": None, "board": snapshot(obs)}]
    guard = 0
    while not obs["done"] and guard < 25000:
        guard += 1
        acting = int(env.runner.getActiveTribeID())
        if bool(env.runner.activeSeatIsBot()):
            desc = [int(v) for v in env.runner.stepBot()]
            value = None
            obs = env._observe()
        else:
            grid, scalars, feats = encoder.encode(obs)
            idx, _, value = net.act(grid, scalars, feats,
                                    greedy=seats[acting] == "policy-greedy")
            desc = [int(v) for v in obs["actions"][idx]]
            obs = env.step(idx)
        frame = {"player": acting, "action": action_text(desc, names),
                 "actionDesc": desc, "tick": obs.get("tick"),
                 "scores": obs["scores"]}
        if value is not None:
            frame["value"] = round(value, 3)
        frame["board"] = snapshot(obs) if not obs["done"] else frames[-1]["board"]
        frames.append(frame)

    # Read the final outcome straight from the runner so it's well-defined even
    # if the game hit the step guard without a natural terminal state.
    r = env.runner
    win = [int(r.getWinStatus(i)) for i in range(env.n_players)]
    scores = [int(r.getScore(i)) for i in range(env.n_players)]
    data = {
        "id": req.get("id"), "tribes": tribes, "seats": seats,
        "boardSize": env.board_size, "mode": mode, "maxTicks": max_ticks,
        "names": names,
        "result": {"win": win, "scores": scores, "ticks": int(r.getTick())},
        "frames": frames,
    }
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"game {req.get('id')}: {len(frames)} frames, win={win}, scores={scores}")


if __name__ == "__main__":
    main()
