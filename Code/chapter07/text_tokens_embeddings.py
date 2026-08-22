"""Глава 7: Текст, токены и Embeddings.

Чистый Python-пример проекта Almaz_AI.

Скрипт показывает:
- простой Tokenizer;
- Vocabulary;
- Token IDs;
- <UNK> и <PAD>;
- Padding;
- Attention Mask;
- nn.Embedding;
- Shapes текстового Batch.
"""

import re

import torch
import torch.nn as nn


RANDOM_SEED = 42
SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
EMBEDDING_DIM = 8


def simple_tokenize(text: str) -> list[str]:
    """Простейшая учебная токенизация по словам."""

    text = text.lower()

    return re.findall(
        r"[а-яёa-z0-9]+",
        text,
    )


def build_vocabulary(
    sentences: list[str],
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Создаёт Vocabulary и отображения Token ↔ ID."""

    tokens: list[str] = []

    for sentence in sentences:
        tokens.extend(
            simple_tokenize(sentence)
        )

    vocabulary = (
        SPECIAL_TOKENS
        + sorted(set(tokens))
    )

    token_to_id = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    id_to_token = {
        index: token
        for token, index in token_to_id.items()
    }

    return vocabulary, token_to_id, id_to_token


def encode(
    text: str,
    token_to_id: dict[str, int],
) -> list[int]:
    """Преобразует текст в Token IDs."""

    unknown_id = token_to_id["<UNK>"]

    return [
        token_to_id.get(token, unknown_id)
        for token in simple_tokenize(text)
    ]


def pad_sequences(
    sequences: list[list[int]],
    pad_id: int,
) -> torch.Tensor:
    """Добавляет Padding и возвращает Batch Tensor."""

    max_length = max(
        len(sequence)
        for sequence in sequences
    )

    padded = [
        sequence
        + [pad_id]
        * (max_length - len(sequence))
        for sequence in sequences
    ]

    return torch.tensor(
        padded,
        dtype=torch.long,
    )


def create_attention_mask(
    token_batch: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Создаёт 1 для Tokens и 0 для Padding."""

    return (
        token_batch != pad_id
    ).long()


def main() -> None:
    """Запускает учебный текстовый pipeline."""

    torch.manual_seed(RANDOM_SEED)

    sentences = [
        "я люблю python",
        "я изучаю нейросети",
        "python помогает изучать ии",
        "нейросети работают с тензорами",
    ]

    (
        vocabulary,
        token_to_id,
        id_to_token,
    ) = build_vocabulary(sentences)

    print("=== VOCABULARY ===")

    for token_id, token in id_to_token.items():
        print(token_id, "→", token)

    print("\nVocabulary size:", len(vocabulary))

    encoded = [
        encode(sentence, token_to_id)
        for sentence in sentences
    ]

    print("\n=== TOKEN IDS ===")

    for sentence, ids in zip(
        sentences,
        encoded,
    ):
        print(sentence, "→", ids)

    pad_id = token_to_id["<PAD>"]

    token_batch = pad_sequences(
        encoded,
        pad_id,
    )

    attention_mask = create_attention_mask(
        token_batch,
        pad_id,
    )

    print("\n=== BATCH ===")
    print(token_batch)
    print("Shape:", tuple(token_batch.shape))

    print("\n=== ATTENTION MASK ===")
    print(attention_mask)

    embedding = nn.Embedding(
        num_embeddings=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        padding_idx=pad_id,
    )

    embedded = embedding(token_batch)

    print("\n=== EMBEDDINGS ===")
    print(
        "Embedding Matrix:",
        tuple(embedding.weight.shape),
    )

    print(
        "Embedded Batch:",
        tuple(embedded.shape),
    )

    sample_token = "python"
    sample_id = token_to_id[sample_token]

    print(
        f"\nToken '{sample_token}' ID:",
        sample_id,
    )

    print(
        "Vector:",
        embedding.weight[sample_id],
    )

    print("\nГлавный pipeline:")
    print(
        "Text → Tokens → Token IDs → Padding → "
        "Embedding → [Batch, Sequence, Embedding Dim]"
    )


if __name__ == "__main__":
    main()
