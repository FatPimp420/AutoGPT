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
BOT_SEATS = {"simple", "random", "donothing"}


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
    max_ticks = int(req.get("max_ticks", 12))
    seed = int(req.get("seed") or time.time())
    rng = np.random.default_rng(seed)

    bot_types = [s if s in BOT_SEATS else "" for s in seats]
    env = TribesEnv(tribes=tribes, game_mode=mode, bot_types=bot_types,
                    max_ticks=max_ticks)
    env.runner.setAutoPlayBots(False)
    obs = env.reset(rng.integers(1 << 30), rng.integers(1 << 30))
    encoder = Encoder(env.type_sizes, env.board_size,
                      max_ticks if mode == "CAPITALS" else 30)

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

    data = {
        "id": req.get("id"), "tribes": tribes, "seats": seats,
        "boardSize": env.board_size, "mode": mode, "maxTicks": max_ticks,
        "names": names,
        "result": {"win": obs["win"], "scores": obs["scores"], "ticks": obs["tick"]},
        "frames": frames,
    }
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"game {req.get('id')}: {len(frames)} frames, win={obs['win']}, "
          f"scores={obs['scores']}")


if __name__ == "__main__":
    main()
