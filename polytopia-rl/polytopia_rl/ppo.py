"""PPO with GAE over ragged (variable-size) legal-action sets.

The buffer stores one entry per decision point (any player: self-play shares the
policy across seats). Action feature matrices vary in length; minibatches pad to
the batch max and mask. GAE runs per (episode, player) trajectory since each
player only observes the game at their own decision points.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Step:
    grid: np.ndarray
    scalars: np.ndarray
    action_feats: np.ndarray
    action_idx: int
    logp: float
    value: float
    reward: float = 0.0  # filled in when the next decision point / terminal is known


@dataclass
class Trajectory:
    steps: list = field(default_factory=list)


def compute_gae(traj, gamma=0.999, lam=0.95):
    """Returns (advantages, returns) for one player trajectory (terminal after last step)."""
    n = len(traj.steps)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = traj.steps[t + 1].value if t + 1 < n else 0.0
        delta = traj.steps[t].reward + gamma * next_value - traj.steps[t].value
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    returns = adv + np.array([s.value for s in traj.steps], dtype=np.float32)
    return adv, returns


class PPOTrainer:
    def __init__(self, net, lr=3e-4, clip=0.2, epochs=3, minibatch_size=256,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 gamma=0.999, lam=0.95):
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.clip = clip
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam

    def update(self, trajectories):
        steps, advs, rets = [], [], []
        for traj in trajectories:
            if not traj.steps:
                continue
            a, r = compute_gae(traj, self.gamma, self.lam)
            steps.extend(traj.steps)
            advs.append(a)
            rets.append(r)
        if not steps:
            return {}
        advs = np.concatenate(advs)
        rets = np.concatenate(rets)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        n = len(steps)
        idx_all = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        batches = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch_size):
                mb = idx_all[start:start + self.minibatch_size]
                if len(mb) < 2:
                    continue
                batch = [steps[i] for i in mb]
                max_a = max(s.action_feats.shape[0] for s in batch)
                d = batch[0].action_feats.shape[1]
                feats = np.zeros((len(mb), max_a, d), dtype=np.float32)
                mask = np.zeros((len(mb), max_a), dtype=bool)
                for j, s in enumerate(batch):
                    k = s.action_feats.shape[0]
                    feats[j, :k] = s.action_feats
                    mask[j, :k] = True

                grid = torch.as_tensor(np.stack([s.grid for s in batch]))
                scal = torch.as_tensor(np.stack([s.scalars for s in batch]))
                feats_t = torch.as_tensor(feats)
                mask_t = torch.as_tensor(mask)
                acts = torch.as_tensor([s.action_idx for s in batch])
                old_logp = torch.as_tensor([s.logp for s in batch], dtype=torch.float32)
                mb_adv = torch.as_tensor(advs[mb], dtype=torch.float32)
                mb_ret = torch.as_tensor(rets[mb], dtype=torch.float32)

                logits, values = self.net(grid, scal, feats_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts)
                ratio = torch.exp(logp - old_logp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = ((values - mb_ret) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
                stats["clip_frac"] += float(((ratio - 1).abs() > self.clip).float().mean())
                batches += 1

        return {k: v / max(1, batches) for k, v in stats.items()} | {"steps": n}
