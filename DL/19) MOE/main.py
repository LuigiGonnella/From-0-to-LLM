import torch
from transformer import MOETransformerBlock
import torch.nn.functional as F
 
# example 
d_model = 32
d_ff = 64
num_experts = 4
top_k = 2
lambda_bal = 0.01
seq_len = 5

model = MOETransformerBlock(d_model, seq_len, d_ff, num_experts=num_experts, top_k=top_k, lambda_bal=lambda_bal)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# fake data
B, T, D = 2, 5, d_model
x = torch.randn(B, T, D)
y_true = torch.randn(B, T, D)

for step in range(100):
    optimizer.zero_grad()
    y_pred, bal_loss = model(x)
    # task loss: MSE 
    task_loss = F.mse_loss(y_pred, y_true)
    # totale
    loss = task_loss + bal_loss
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | Task: {task_loss.item():.4f} | Bal: {bal_loss.item():.4f}")
