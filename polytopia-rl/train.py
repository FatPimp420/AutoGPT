"""Self-play PPO training for Tribes via the JPype bridge.

One policy network controls both tribes (parameter-shared self-play). Each
player's decision points form a separate trajectory; rewards are a dense
score-difference shaping plus a terminal win/loss bonus.

Usage:
  .venv/bin/python train.py --iterations 3 --rollout-steps 2048 --max-ticks 12
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from polytopia_rl.bridge import TribesEnv
from polytopia_rl.encoder import Encoder
from polytopia_rl.model import PolicyValueNet
from polytopia_rl.ppo import PPOTrainer, Step, Trajectory

ROOT = Path(__file__).resolve().parent


def play_selfplay_game(env, encoder, net, rng, shaping_coef, win_reward,
                       capture_reward=0.0, kill_reward=0.0):
    """Plays one self-play game; returns (per-player trajectories, game info).

    Reward = terminal win/loss (win_reward) + dense score-diff shaping
    (shaping_coef) + per-event bonuses for gaining a city (capture_reward, also
    penalising a lost city) and for unit kills (kill_reward)."""
    obs = env.reset(rng.integers(1 << 30), rng.integers(1 << 30))
    n = env.n_players
    trajs = {p: Trajectory() for p in range(n)}
    prev_diff = {p: 0.0 for p in range(n)}
    steps = 0

    # progression tracking (sampled once per game turn, not per action)
    first_capture, first_elim = -1, -1        # tick of first city gain / elimination
    _, _, prev_stars = env.tick_stats()
    stars_gen = 0                             # total stars earned across seats (proxy)
    prev_counts = env.player_counts()         # [cities, kills, alive] per player
    event_shaping = capture_reward or kill_reward
    prev_tick = obs["tick"]

    while not obs["done"]:
        p = obs["player"]
        diff = obs["scores"][p] - max(obs["scores"][q] for q in range(n) if q != p)
        if trajs[p].steps:  # shaping reward for this player's previous action
            trajs[p].steps[-1].reward += shaping_coef * (diff - prev_diff[p])
        prev_diff[p] = diff

        grid, scalars, action_feats = encoder.encode(obs)
        idx, logp, value = net.act(grid, scalars, action_feats)
        trajs[p].steps.append(Step(grid, scalars, action_feats, idx, logp, value))
        obs = env.step(idx)
        steps += 1

        if obs.get("tick", prev_tick) != prev_tick and not obs["done"]:
            prev_tick = obs["tick"]
            cities, alive, stars = env.tick_stats()
            stars_gen += max(0, stars - prev_stars)
            prev_stars = stars
            if first_capture < 0 and cities > n:
                first_capture = prev_tick
            if first_elim < 0 and alive < n:
                first_elim = prev_tick
            if event_shaping:  # attribute city/kill deltas to each seat's last action
                counts = env.player_counts()
                for q in range(n):
                    if not trajs[q].steps:
                        continue
                    d_city = counts[q][0] - prev_counts[q][0]
                    d_kill = counts[q][1] - prev_counts[q][1]
                    trajs[q].steps[-1].reward += (capture_reward * d_city
                                                  + kill_reward * d_kill)
                prev_counts = counts

    for p in range(n):
        if trajs[p].steps:
            trajs[p].steps[-1].reward += win_reward * obs["win"][p]

    # snapshot the winner (decisive winner, else the top-scoring seat)
    win = obs["win"]
    winner = win.index(1) if 1 in win else int(np.argmax(obs["scores"]))
    w = env.player_stats(winner)
    _, alive_final, _ = env.tick_stats()
    tiles = env.board_size * env.board_size
    info = {
        "steps": steps, "ticks": obs["tick"], "scores": obs["scores"], "win": win,
        "decisive": 1 if 1 in win else 0,
        "winner_cities": w["cities"], "winner_techs": w["techs"],
        "winner_kills": w["kills"], "winner_stars": w["stars"],
        "map_control": round(w["tiles"] / tiles, 4),
        "elims": n - alive_final,
        "stars_gen": stars_gen,
        "first_capture": first_capture, "first_elim": first_elim,
    }
    return list(trajs.values()), info


def evaluate_vs_bot(encoder, net, bot, games, max_ticks, mode, rng):
    """Greedy policy in seat 0 vs a built-in Java bot in seat 1. Returns win rate."""
    env = TribesEnv(game_mode=mode, bot_types=["", bot], max_ticks=max_ticks)
    wins = 0
    for _ in range(games):
        obs = env.reset(rng.integers(1 << 30), rng.integers(1 << 30))
        while not obs["done"]:
            grid, scalars, action_feats = encoder.encode(obs)
            idx, _, _ = net.act(grid, scalars, action_feats, greedy=True)
            obs = env.step(idx)
        wins += obs["win"][0] == 1
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--max-ticks", type=int, default=12,
                    help="turn limit per game (CAPITALS mode); default short for fast iterations")
    ap.add_argument("--mode", default="CAPITALS", choices=["CAPITALS", "SCORE"])
    ap.add_argument("--shaping-coef", type=float, default=1e-3)
    ap.add_argument("--win-reward", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=0,
                    help="if >0, evaluate greedy policy vs the random bot after training")
    ap.add_argument("--save", default=str(ROOT / "checkpoints" / "policy.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = TribesEnv(game_mode=args.mode, max_ticks=args.max_ticks)
    obs = env.reset(1, 1)  # sizes needed to build encoder/net
    max_ticks = args.max_ticks if args.mode == "CAPITALS" else 30
    encoder = Encoder(env.type_sizes, env.board_size, max_ticks)
    net = PolicyValueNet(encoder.grid_channels, encoder.scalar_dim,
                         encoder.action_dim, env.board_size)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"board={env.board_size}x{env.board_size} grid_channels={encoder.grid_channels} "
          f"action_dim={encoder.action_dim} params={n_params:,}")

    trainer = PPOTrainer(net, lr=args.lr)

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        trajs, games, steps = [], [], 0
        while steps < args.rollout_steps:
            game_trajs, info = play_selfplay_game(env, encoder, net, rng,
                                                  args.shaping_coef, args.win_reward)
            trajs.extend(game_trajs)
            games.append(info)
            steps += info["steps"]
        collect_s = time.time() - t0

        t1 = time.time()
        stats = trainer.update(trajs)
        update_s = time.time() - t1

        mean_len = np.mean([g["steps"] for g in games])
        mean_ticks = np.mean([g["ticks"] for g in games])
        mean_score = np.mean([g["scores"] for g in games])
        decided = sum(1 for g in games if 1 in g["win"])
        print(f"iter {it:3d} | games {len(games):3d} (decided {decided}) | steps {steps:5d} "
              f"| len {mean_len:6.1f} ticks {mean_ticks:4.1f} score {mean_score:7.1f} "
              f"| pi {stats['policy_loss']:+.4f} vf {stats['value_loss']:.4f} "
              f"ent {stats['entropy']:.3f} clip {stats['clip_frac']:.3f} "
              f"| collect {collect_s:.1f}s update {update_s:.1f}s")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(),
                "config": {"grid_channels": encoder.grid_channels,
                           "scalar_dim": encoder.scalar_dim,
                           "action_dim": encoder.action_dim,
                           "board_size": env.board_size}}, save_path)
    print(f"saved checkpoint to {save_path}")

    if args.eval_games > 0:
        wr = evaluate_vs_bot(encoder, net, "random", args.eval_games,
                             args.max_ticks, args.mode, rng)
        print(f"eval vs random bot: win rate {wr:.2f} over {args.eval_games} games")


if __name__ == "__main__":
    main()
