"""Long-running, externally controllable self-play PPO trainer.

Reads runs/control.json before every iteration, so training can be paused,
resumed, and re-ruled mid-run without losing the network:

{
  "paused": false,
  "mode": "CAPITALS",          // or "SCORE"
  "max_ticks": 12,             // CAPITALS turn limit (SCORE is fixed at 30)
  "rollout_steps": 2048,
  "shaping_coef": 0.001,       // dense score-difference reward
  "win_reward": 1.0,           // terminal +/- bonus
  "lr": 3e-4,
  "ent_coef": 0.01,
  "note": "free-text shown on the dashboard"
}

Outputs under runs/:
  metrics.jsonl     one line per iteration (append-only)
  status.json       current state for the dashboard
  events.jsonl      config changes / pauses (the "rule change" history)
  replay.json       latest recorded self-play game (refreshed periodically)
  publish.log       one line per dashboard-refresh request (watched externally)
Checkpoints go to checkpoints/policy.pt every SAVE_EVERY iterations.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from polytopia_rl.bridge import TribesEnv
from polytopia_rl.encoder import Encoder
from polytopia_rl.model import PolicyValueNet
from polytopia_rl.ppo import PPOTrainer
from train import play_selfplay_game

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
CKPT = ROOT / "checkpoints" / "policy.pt"
SAVE_EVERY = 25          # iterations between checkpoint + replay refresh
PUBLISH_EVERY = 150      # iterations between dashboard publish requests

DEFAULT_CONTROL = {
    "paused": False, "mode": "CAPITALS", "max_ticks": 12, "rollout_steps": 2048,
    "shaping_coef": 1e-3, "win_reward": 1.0, "lr": 3e-4, "ent_coef": 0.01,
    "note": "default rules",
}
ENV_KEYS = ("mode", "max_ticks")


def read_control():
    path = RUNS / "control.json"
    try:
        cfg = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}
    merged = DEFAULT_CONTROL | cfg
    if not path.exists():
        path.write_text(json.dumps(merged, indent=2))
    return merged


def log_line(name, obj):
    with open(RUNS / name, "a") as f:
        f.write(json.dumps(obj) + "\n")


def write_status(state, it, cfg, extra=None):
    obj = {"state": state, "iteration": it, "config": cfg,
           "updated": time.time()} | (extra or {})
    (RUNS / "status.json").write_text(json.dumps(obj))


def record_replay(env, encoder, net, rng, out):
    """Self-play game with the current net, saved for the dashboard viewer."""
    import record_game as rg
    from polytopia_rl.bridge import start_jvm
    start_jvm()
    names = {k: rg.enum_names(v) for k, v in {
        "action": "core.Types$ACTION", "terrain": "core.Types$TERRAIN",
        "resource": "core.Types$RESOURCE", "building": "core.Types$BUILDING",
        "unit": "core.Types$UNIT", "tech": "core.Types$TECHNOLOGY"}.items()}
    obs = env.reset(rng.integers(1 << 30), rng.integers(1 << 30))
    frames = [{"tick": obs["tick"], "player": obs["player"], "scores": obs["scores"],
               "action": None, "board": rg.snapshot(obs)}]
    last = None
    while not obs["done"]:
        grid, scalars, feats = encoder.encode(obs)
        idx, _, value = net.act(grid, scalars, feats)
        desc, acting = obs["actions"][idx], obs["player"]
        last = obs
        obs = env.step(idx)
        frames.append({"player": acting, "action": rg.action_text(desc, names),
                       "value": round(value, 3), "actionDesc": [int(v) for v in desc],
                       "tick": obs["tick"], "scores": obs["scores"],
                       "board": rg.snapshot(obs if not obs["done"] else last)})
    data = {"tribes": env.tribes, "boardSize": env.board_size,
            "maxTicks": int(env.runner.getTick()), "names": names,
            "result": {"win": obs["win"], "scores": obs["scores"], "ticks": obs["tick"]},
            "frames": frames}
    Path(out).write_text(json.dumps(data, separators=(",", ":")))


def build(cfg, net=None):
    """(Re)builds env/encoder/net/trainer for a config; keeps net if given."""
    env = TribesEnv(game_mode=cfg["mode"], max_ticks=cfg["max_ticks"])
    env.reset(1, 1)
    max_ticks = cfg["max_ticks"] if cfg["mode"] == "CAPITALS" else 30
    encoder = Encoder(env.type_sizes, env.board_size, max_ticks)
    if net is None:
        net = PolicyValueNet(encoder.grid_channels, encoder.scalar_dim,
                             encoder.action_dim, env.board_size)
        if CKPT.exists():
            net.load_state_dict(torch.load(CKPT, weights_only=True)["model"])
            print(f"resumed weights from {CKPT}")
    trainer = PPOTrainer(net, lr=cfg["lr"], ent_coef=cfg["ent_coef"])
    return env, encoder, net, trainer


def save_ckpt(net, encoder, env, it):
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"model": net.state_dict(), "iteration": it,
                "config": {"grid_channels": encoder.grid_channels,
                           "scalar_dim": encoder.scalar_dim,
                           "action_dim": encoder.action_dim,
                           "board_size": env.board_size}}, CKPT)


def main():
    RUNS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(int(time.time()))

    cfg = read_control()
    env, encoder, net, trainer = build(cfg)
    it = 0
    if CKPT.exists():
        it = int(torch.load(CKPT, weights_only=True).get("iteration", 0))
    log_line("events.jsonl", {"t": time.time(), "event": "daemon started",
                              "iteration": it, "config": cfg})
    print(f"daemon up at iteration {it}, config: {cfg}")

    while True:
        new_cfg = read_control()
        if new_cfg != cfg:
            changed = {k: v for k, v in new_cfg.items() if cfg.get(k) != v}
            log_line("events.jsonl", {"t": time.time(), "event": "rules changed",
                                      "iteration": it, "changed": changed})
            if any(cfg.get(k) != new_cfg.get(k) for k in ENV_KEYS):
                env, encoder, net, trainer = build(new_cfg, net)  # keep weights
            else:
                for g in trainer.opt.param_groups:
                    g["lr"] = new_cfg["lr"]
                trainer.ent_coef = new_cfg["ent_coef"]
            cfg = new_cfg
            print(f"rules changed: {changed}")

        if cfg["paused"]:
            write_status("paused", it, cfg)
            time.sleep(5)
            continue

        t0 = time.time()
        trajs, games, steps = [], [], 0
        while steps < cfg["rollout_steps"]:
            game_trajs, info = play_selfplay_game(
                env, encoder, net, rng, cfg["shaping_coef"], cfg["win_reward"])
            trajs.extend(game_trajs)
            games.append(info)
            steps += info["steps"]
        stats = trainer.update(trajs)
        it += 1

        row = {
            "iter": it, "t": time.time(), "games": len(games), "steps": steps,
            "decided": sum(1 for g in games if 1 in g["win"]),
            "mean_len": round(float(np.mean([g["steps"] for g in games])), 1),
            "mean_ticks": round(float(np.mean([g["ticks"] for g in games])), 1),
            "mean_score": round(float(np.mean([g["scores"] for g in games])), 1),
            "policy_loss": round(stats["policy_loss"], 5),
            "value_loss": round(stats["value_loss"], 5),
            "entropy": round(stats["entropy"], 4),
            "clip_frac": round(stats["clip_frac"], 4),
            "iter_s": round(time.time() - t0, 1),
            "mode": cfg["mode"], "max_ticks": cfg["max_ticks"],
        }
        log_line("metrics.jsonl", row)
        write_status("training", it, cfg, {"last_iter": row})
        print(f"iter {it} | {len(games)} games | ent {row['entropy']:.3f} "
              f"| vf {row['value_loss']:.3f} | {row['iter_s']}s")

        if it % SAVE_EVERY == 0:
            save_ckpt(net, encoder, env, it)
            try:
                record_replay(env, encoder, net, rng, RUNS / "replay.json")
            except Exception as e:  # replay is cosmetic; never kill training for it
                log_line("events.jsonl", {"t": time.time(),
                                          "event": f"replay failed: {e}"})
        if it % PUBLISH_EVERY == 0:
            log_line("publish.log", {"t": time.time(), "iteration": it})


if __name__ == "__main__":
    main()
