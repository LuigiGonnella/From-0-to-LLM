# Mixture of Experts (MoE)

This folder contains an implementation of the Mixture of Experts architecture, a scalable approach for training large neural networks by selectively activating different experts.

## Files

- **moe.py**: Core MoE implementation
  - `Expert`: A simple feed-forward network module
  - `MoELayer`: Routing mechanism that selects top-k experts and includes load balancing loss
  
- **transformer.py**: Transformer blocks with attention mechanisms
  - `ScaledDotProductAttention`: Standard attention computation
  - `AttentionHead`: Multi-head attention implementation
  - `MOETransformerBlock`: Complete transformer block combining attention with MoE layers

- **main.py**: Training example demonstrating how to train the model on synthetic data

## Usage

Run the training example:
```bash
python main.py
```

The model trains on synthetic data and monitors task loss and load-balancing loss during training.

## Key Concepts

- **Router**: Assigns input tokens to multiple experts based on learned routing weights
- **Top-k Selection**: Each token is processed by the top-k most relevant experts
- **Load Balancing Loss**: Ensures even distribution of tokens across experts to prevent collapse
