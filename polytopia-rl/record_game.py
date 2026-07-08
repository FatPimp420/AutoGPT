"""Records a game played by a trained checkpoint as JSON for the replay viewer.

Self-play (both seats driven by the checkpoint, sampled) so every action passes
through Python and each board state can be snapshotted.

Usage: .venv/bin/python record_game.py [--out replay.json] [--max-ticks 12] [--seed 7]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from polytopia_rl.bridge import TribesEnv, start_jvm
from polytopia_rl.encoder import Encoder
from polytopia_rl.model import PolicyValueNet

ROOT = Path(__file__).resolve().parent

# grid channels worth shipping to the viewer (see RLGameRunner.getObservation)
VIEW_CHANNELS = {
    "terrain": 0, "resource": 1, "building": 2, "unitType": 3, "unitOwner": 4,
    "unitHp": 5, "unitMaxHp": 6, "cityOwner": 9, "cityLevel": 10, "road": 11,
}


def downsample_frames(frames, board_size, budget_kb=1400):
    """Caps a replay's JSON size by keeping evenly-spaced frames (always the
    first and last). Each frame stores board_size^2 * 10 channels, so long or
    large-map games are trimmed to keep the file viewable and pushable."""
    if len(frames) <= 2:
        return frames
    per_frame_kb = 3.46 * (board_size / 11.0) ** 2  # measured at 11x11
    max_frames = max(2, int(budget_kb / max(per_frame_kb, 0.1)))
    if len(frames) <= max_frames:
        return frames
    n = len(frames)
    idx = sorted(set([0] + [round(i * (n - 1) / (max_frames - 1))
                            for i in range(max_frames)] + [n - 1]))
    return [frames[i] for i in idx]


def enum_names(java_enum):
    import jpype
    cls = jpype.JClass(java_enum)
    return [str(v) for v in cls.values()]


def action_text(desc, names):
    t = names["action"][desc[0]]
    src = f"({desc[1]},{desc[2]})" if desc[1] >= 0 else ""
    tgt = f"({desc[3]},{desc[4]})" if desc[3] >= 0 else ""
    extra = ""
    if desc[5] >= 0:
        if t == "BUILD":            extra = names["building"][desc[5]]
        elif t == "SPAWN":          extra = names["unit"][desc[5]]
        elif t == "RESEARCH_TECH":  extra = names["tech"][desc[5]]
        elif t == "RESOURCE_GATHERING": extra = names["resource"][desc[5]]
        else:                       extra = str(desc[5])
    parts = [t]
    if extra: parts.append(extra)
    if src:   parts.append(src)
    if tgt:   parts.append("→" + tgt)
    return " ".join(parts)


def snapshot(obs):
    return {k: obs["grid"][ch].tolist() for k, ch in VIEW_CHANNELS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "policy.pt"))
    ap.add_argument("--out", default=str(ROOT / "replay.json"))
    ap.add_argument("--max-ticks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env = TribesEnv(max_ticks=args.max_ticks)
    rng = np.random.default_rng(args.seed)
    obs = env.reset(rng.integers(1 << 30), rng.integers(1 << 30))
    encoder = Encoder(env.type_sizes, env.board_size, args.max_ticks)

    ckpt = torch.load(args.ckpt, weights_only=True)
    net = PolicyValueNet(**ckpt["config"])
    net.load_state_dict(ckpt["model"])
    net.eval()

    start_jvm()
    names = {
        "action":   enum_names("core.Types$ACTION"),
        "terrain":  enum_names("core.Types$TERRAIN"),
        "resource": enum_names("core.Types$RESOURCE"),
        "building": enum_names("core.Types$BUILDING"),
        "unit":     enum_names("core.Types$UNIT"),
        "tech":     enum_names("core.Types$TECHNOLOGY"),
    }

    frames = [{"tick": obs["tick"], "player": obs["player"], "scores": obs["scores"],
               "action": None, "board": snapshot(obs)}]
    last = None
    while not obs["done"]:
        grid, scalars, feats = encoder.encode(obs)
        idx, _, value = net.act(grid, scalars, feats)
        desc = obs["actions"][idx]
        acting = obs["player"]
        txt = action_text(desc, names)
        last = obs
        obs = env.step(idx)
        frame = {"player": acting, "action": txt, "value": round(value, 3),
                 "actionDesc": [int(v) for v in desc]}
        if obs["done"]:
            frame.update(tick=obs["tick"], scores=obs["scores"], board=snapshot(last))
        else:
            frame.update(tick=obs["tick"], scores=obs["scores"], board=snapshot(obs))
        frames.append(frame)

    data = {
        "tribes": env.tribes, "boardSize": env.board_size, "maxTicks": args.max_ticks,
        "names": names, "result": {"win": obs["win"], "scores": obs["scores"], "ticks": obs["tick"]},
        "frames": downsample_frames(frames, env.board_size),
    }
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"recorded {len(frames)} frames, result win={obs['win']} "
          f"scores={obs['scores']} ticks={obs['tick']} -> {args.out}")


if __name__ == "__main__":
    main()
