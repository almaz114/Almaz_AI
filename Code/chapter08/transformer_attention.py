"""Глава 8: Transformer и Attention.

Чистый Python-пример проекта Almaz_AI.

Скрипт показывает:
- Query, Key, Value;
- Scaled Dot-Product Attention;
- Attention Weights;
- Causal Mask;
- nn.MultiheadAttention;
- простой Transformer Block.
"""

import math

import torch
import torch.nn as nn


RANDOM_SEED = 42
EMBEDDING_DIM = 8
NUM_HEADS = 2
SEQUENCE_LENGTH = 4


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Вычисляет Scaled Dot-Product Attention."""

    d_k = key.shape[-1]

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

    weights = torch.softmax(
        scores,
        dim=-1,
    )

    output = weights @ value

    return output, weights


class TinyTransformerBlock(nn.Module):
    """Упрощённый Transformer Block."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_HEADS,
        ff_dim: int = 16,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                ff_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                ff_dim,
                embedding_dim,
            ),
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Выполняет Attention + FFN + Residual Connections."""

        attention_output, weights = self.attention(
            x,
            x,
            x,
        )

        x = self.norm1(
            x + attention_output
        )

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + feed_forward_output
        )

        return x, weights


def main() -> None:
    """Запускает учебный Attention pipeline."""

    torch.manual_seed(RANDOM_SEED)

    x = torch.randn(
        1,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
    )

    print("Input shape:", tuple(x.shape))

    query_layer = nn.Linear(
        EMBEDDING_DIM,
        EMBEDDING_DIM,
        bias=False,
    )

    key_layer = nn.Linear(
        EMBEDDING_DIM,
        EMBEDDING_DIM,
        bias=False,
    )

    value_layer = nn.Linear(
        EMBEDDING_DIM,
        EMBEDDING_DIM,
        bias=False,
    )

    query = query_layer(x)
    key = key_layer(x)
    value = value_layer(x)

    output, weights = scaled_dot_product_attention(
        query,
        key,
        value,
    )

    print("\n=== SELF-ATTENTION ===")
    print("Q:", tuple(query.shape))
    print("K:", tuple(key.shape))
    print("V:", tuple(value.shape))
    print("Weights:", tuple(weights.shape))
    print("Output:", tuple(output.shape))

    causal_mask = torch.triu(
        torch.ones(
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
        ),
        diagonal=1,
    ).bool()

    causal_output, causal_weights = scaled_dot_product_attention(
        query,
        key,
        value,
        mask=causal_mask,
    )

    print("\n=== CAUSAL ATTENTION ===")
    print(causal_weights)

    block = TinyTransformerBlock()

    block_output, block_weights = block(x)

    print("\n=== TRANSFORMER BLOCK ===")
    print("Input:", tuple(x.shape))
    print("Output:", tuple(block_output.shape))
    print(
        "Attention weights:",
        tuple(block_weights.shape),
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in block.parameters()
    )

    print("Parameters:", total_parameters)

    print("\nГлавный pipeline:")
    print(
        "Embeddings → Q/K/V → Scores → Scale → "
        "Softmax → Weighted Values → Context"
    )


if __name__ == "__main__":
    main()
