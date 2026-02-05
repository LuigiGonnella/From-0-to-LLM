import torch.nn as nn
import torch
from moe import MoELayer

# Define the Scaled Dot-Product Attention Mechanism
class ScaledDotProductAttention(nn.Module):
    def __init__(self, seq_len, d_k, other_positional_encodings_present):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k
        self.seq_len = seq_len
        self.other_positional_encodings_present = other_positional_encodings_present

        self.rel_layer = torch.randn(1, self.seq_len, self.seq_len, requires_grad=True) # Create a learnable relative positional encoding

    def forward(self, query, key, value):
        # Compute the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)) # Scaled Dot-Product
        
        # Add relative positional encodings if no other positional encodings are present
        if not self.other_positional_encodings_present:
            scores = scores + self.rel_layer
        
        # Compute the attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute the output
        output = torch.matmul(attention_weights, value)
        
        return output
    

class AttentionHead(nn.Module):
    def __init__(self, d_model, seq_len, other_positional_encodings_present=False):
        """
        Initializes the AttentionHead.

        Args:
        - d_model (int): The dimensionality of the input and output features.
        - seq_len (int): The length of the input sequence.
        - other_positional_encodings_present (bool): Whether other positional encodings are being used.
        
        Notes:
        - This class uses the Scaled Dot-Product Attention mechanism to compute attention.
        - It also applies linear transformations to the query, key, and value matrices.
        """
        super(AttentionHead, self).__init__()
        self.d_k = d_model
        self.seq_len = seq_len

        self.query_layer = nn.Linear(d_model, self.d_k)
        self.key_layer = nn.Linear(d_model, self.d_k)
        self.value_layer = nn.Linear(d_model, self.d_k)

        self.scaled_dot_attention = ScaledDotProductAttention(self.seq_len, self.d_k, other_positional_encodings_present)

        self.fc_out = nn.Linear(self.d_k, self.d_k)

    def forward(self, query, key, value):
        """
        Computes the attention output and weights.

        Args:
        - query: The query matrix (shape: [batch_size, seq_len, d_model]).
        - key: The key matrix (shape: [batch_size, seq_len, d_model]).
        - value: The value matrix (shape: [batch_size, seq_len, d_model]).
        
        Returns:
        - output: The final output after applying attention.
        - attention_weights: The attention weights for each query/key pair.
        """
        
        Q = self.query_layer(query) #we pass the same inputs to all the layers that generates Q,K and V
        K = self.key_layer(key)
        V = self.value_layer(value)

        attention_output = self.scaled_dot_attention(Q, K, V)

        attention_output = attention_output.squeeze(1) #eliminates dimensions of size 1
        output = self.fc_out(attention_output)

        return output
    


class MOETransformerBlock(nn.Module):
    def __init__(self, d_model, seq_len, d_ff, num_experts=4, top_k=1, lambda_bal=0.01):
        super().__init__()
        self.attn = AttentionHead(d_model, seq_len)
        self.moe = MoELayer(d_model, d_ff, num_experts=num_experts, top_k=top_k, lambda_bal=lambda_bal)

    def forward(self, x):
        x = x + self.attn(x,x,x)
        moe_out, bal_loss = self.moe(x)
        x = x + moe_out
        return x, bal_loss