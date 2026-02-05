import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=1, lambda_bal=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_bal = lambda_bal

        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        logits = self.router(x)                  # (B, T, E)
        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # (B, T, K)
        topk_probs = F.softmax(topk_logits, dim=-1)                     # (B, T, K)

        output = torch.zeros_like(x)

        # scatter to experts
        for k in range(self.top_k):
            expert_idx = topk_idx[..., k]                  # (B, T)
            expert_weight = topk_probs[..., k]            # (B, T)
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i)
                if mask.any():
                    x_masked = x[mask]                   # (N_i, D)
                    w_masked = expert_weight[mask].unsqueeze(-1)  # (N_i, 1)
                    output[mask] += expert(x_masked) * w_masked

        # Load balancing loss
        # importance = mean probability per expert
        importance = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            importance.scatter_add_(0, topk_idx[..., k].view(-1), topk_probs[..., k].view(-1))
        importance = importance / (B * T)
        
        # load = fraction of tokens per expert
        load = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            load[i] = (topk_idx == i).float().sum() / (B*T)
        bal_loss = (importance * load).sum() * self.num_experts

        return output, self.lambda_bal * bal_loss
