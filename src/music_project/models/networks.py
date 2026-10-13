"""LSTM and pre-LayerNorm causal Transformer with rotary positions and SwiGLU."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LSTMModel(nn.Module):
    def __init__(
        self, vocab_size: int, pad_id: int, width: int = 256, layers: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            width, width, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )
        self.head = nn.Linear(width, vocab_size)

    def forward(self, input_ids, attention_mask=None, *, last_only=False):
        # Hidden state starts fresh for each independent performance crop.
        hidden, _ = self.lstm(self.dropout(self.embedding(input_ids)))
        if last_only:
            hidden = hidden[:, -1:]
        return self.head(self.dropout(hidden))


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("Width must divide into even-sized rotary attention heads")
        self.heads, self.head_width = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.dropout = dropout
        self.register_buffer(
            "inv_freq",
            1
            / (
                10_000
                ** (torch.arange(0, self.head_width, 2, dtype=torch.float32) / self.head_width)
            ),
            persistent=False,
        )

    def rotary(self, tensor):
        positions = torch.arange(tensor.shape[-2], device=tensor.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq[None, :]
        cosine, sine = angles.cos().to(tensor.dtype), angles.sin().to(tensor.dtype)
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1)

    def forward(self, hidden, attention_mask):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_width).unbind(2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        q, k = self.rotary(q), self.rotary(k)
        mask = None
        if attention_mask is not None:
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            mask = causal[None, None] & attention_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(attended.transpose(1, 2).contiguous().reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.attention_norm, self.ffn_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, dropout)
        self.gate_up = nn.Linear(width, 2 * feedforward, bias=False)
        self.down = nn.Linear(feedforward, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, attention_mask):
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), attention_mask))
        gate, up = self.gate_up(self.ffn_norm(hidden)).chunk(2, dim=-1)
        return hidden + self.dropout(self.down(F.silu(gate) * up))


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        width=512,
        layers=8,
        heads=8,
        feedforward=1536,
        dropout=0.1,
        tied_embeddings=True,
        gradient_checkpointing=False,
        context_length=2048,
    ):
        super().__init__()
        self.context_length, self.gradient_checkpointing = context_length, gradient_checkpointing
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(width, heads, feedforward, dropout) for _ in range(layers)]
        )
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, vocab_size, bias=False)
        self.apply(self.initialize)
        if tied_embeddings:
            self.head.weight = self.embedding.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids, attention_mask=None, *, last_only=False):
        if input_ids.shape[1] > self.context_length:
            raise ValueError("Input exceeds checkpoint context length")
        hidden = self.dropout(self.embedding(input_ids))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, attention_mask, use_reentrant=False)
            else:
                hidden = block(hidden, attention_mask)
        if last_only:
            hidden = hidden[:, -1:]
        return self.head(self.norm(hidden))


def build_model(config: dict, vocab_size: int, pad_id: int, context_length: int):
    kind = config["kind"]
    values = {key: value for key, value in config.items() if key != "kind"}
    if kind == "lstm":
        return LSTMModel(vocab_size, pad_id, **values)
    if kind == "transformer":
        return TransformerModel(vocab_size, pad_id, context_length=context_length, **values)
    raise ValueError(f"Unknown model kind: {kind}")
