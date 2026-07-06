"""Policy-value network with per-action scoring.

State: CNN over the encoded board planes + MLP over global scalars -> embedding.
Policy: each legal action's feature vector is embedded and scored against the
state embedding, giving one logit per legal action (categorical over the ragged
legal set — no fixed global action space, no masking of impossible slots).
Value: MLP head on the state embedding.
"""

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, grid_channels, scalar_dim, action_dim, board_size,
                 conv_dim=64, state_dim=256, act_emb_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(grid_channels, conv_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.scalar_mlp = nn.Sequential(nn.Linear(scalar_dim, 64), nn.ReLU())
        self.state_mlp = nn.Sequential(
            nn.Linear(conv_dim * 16 + 64, state_dim), nn.ReLU(),
        )
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, act_emb_dim), nn.ReLU(),
            nn.Linear(act_emb_dim, act_emb_dim), nn.ReLU(),
        )
        self.logit_head = nn.Sequential(
            nn.Linear(state_dim + act_emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def state_embedding(self, grid, scalars):
        x = self.conv(grid).flatten(1)
        s = self.scalar_mlp(scalars)
        return self.state_mlp(torch.cat([x, s], dim=-1))

    def forward(self, grid, scalars, action_feats, action_mask):
        """grid: (B,C,n,n); scalars: (B,S); action_feats: (B,A,D) padded;
        action_mask: (B,A) True where the slot holds a real action.
        Returns logits (B,A) with -inf on padding, and values (B,)."""
        emb = self.state_embedding(grid, scalars)                 # (B, H)
        a = self.action_emb(action_feats)                         # (B, A, E)
        e = emb.unsqueeze(1).expand(-1, a.shape[1], -1)           # (B, A, H)
        logits = self.logit_head(torch.cat([e, a], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        value = self.value_head(emb).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, grid, scalars, action_feats, greedy=False):
        """Single-state action selection. Inputs unbatched numpy-convertible."""
        g = torch.as_tensor(grid).unsqueeze(0)
        s = torch.as_tensor(scalars).unsqueeze(0)
        a = torch.as_tensor(action_feats).unsqueeze(0)
        mask = torch.ones(1, a.shape[1], dtype=torch.bool)
        logits, value = self.forward(g, s, a, mask)
        dist = torch.distributions.Categorical(logits=logits)
        idx = logits.argmax(dim=-1) if greedy else dist.sample()
        return int(idx.item()), float(dist.log_prob(idx).item()), float(value.item())
