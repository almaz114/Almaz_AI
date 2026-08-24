"""Глава 12: Fine-tuning, LoRA и QLoRA.

Чистый учебный пример проекта Almaz_AI.

Скрипт демонстрирует математическую идею LoRA:
- Base Linear заморожен;
- обучаются только матрицы A и B;
- Delta W = B @ A;
- сравнивается количество параметров;
- выполняется маленькое обучение.

Это не Fine-tuning большой LLM, а прозрачная учебная модель LoRA.
"""

from __future__ import annotations

import torch
import torch.nn as nn


RANDOM_SEED = 42
INPUT_DIM = 16
OUTPUT_DIM = 16
RANK = 4
ALPHA = 8.0
LEARNING_RATE = 0.03
EPOCHS = 200


class LoRALinear(nn.Module):
    """Linear-слой с замороженной Base-веткой и LoRA Adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int = RANK,
        alpha: float = ALPHA,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(
                "rank должен быть положительным"
            )

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.randn(
                rank,
                base_layer.in_features,
            ) * 0.01
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Вычисляет Base output + LoRA update."""

        base_output = self.base(
            features
        )

        lora_output = (
            features
            @ self.lora_A.T
            @ self.lora_B.T
        )

        return (
            base_output
            + self.scale * lora_output
        )

    def delta_weight(
        self,
    ) -> torch.Tensor:
        """Возвращает эффективное LoRA-обновление веса."""

        return (
            self.lora_B
            @ self.lora_A
        ) * self.scale


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """Возвращает total и trainable parameters."""

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def create_dataset() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Создаёт маленький искусственный Dataset."""

    features = torch.randn(
        128,
        INPUT_DIM,
    )

    target_projection = torch.randn(
        INPUT_DIM,
        OUTPUT_DIM,
    )

    targets = (
        features
        @ target_projection
    )

    return features, targets


def train_lora(
    model: LoRALinear,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> list[float]:
    """Обучает только LoRA parameters."""

    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
    )

    loss_function = nn.MSELoss()
    history: list[float] = []

    for epoch in range(EPOCHS):
        predictions = model(
            features
        )

        loss = loss_function(
            predictions,
            targets,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            loss.item()
        )

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Loss: {loss.item():.6f}"
            )

    return history


def main() -> None:
    """Запускает учебный LoRA experiment."""

    torch.manual_seed(
        RANDOM_SEED
    )

    base_layer = nn.Linear(
        INPUT_DIM,
        OUTPUT_DIM,
    )

    model = LoRALinear(
        base_layer,
        rank=RANK,
        alpha=ALPHA,
    )

    total, trainable = (
        count_parameters(model)
    )

    print("=== PARAMETERS ===")
    print("Total:", total)
    print("Trainable:", trainable)
    print(
        "Trainable %:",
        round(
            100 * trainable / total,
            3,
        ),
    )

    print("\n=== PARAMETER STATUS ===")

    for name, parameter in (
        model.named_parameters()
    ):
        print(
            name,
            tuple(parameter.shape),
            "requires_grad=",
            parameter.requires_grad,
        )

    base_weight_before = (
        model.base.weight
        .detach()
        .clone()
    )

    adapter_before = (
        model.lora_B
        .detach()
        .clone()
    )

    features, targets = (
        create_dataset()
    )

    print("\n=== TRAIN ===")

    history = train_lora(
        model,
        features,
        targets,
    )

    print("\nFinal Loss:", history[-1])

    print(
        "Base unchanged:",
        torch.allclose(
            base_weight_before,
            model.base.weight,
        ),
    )

    print(
        "LoRA B changed:",
        not torch.allclose(
            adapter_before,
            model.lora_B,
        ),
    )

    print(
        "Delta W shape:",
        tuple(
            model.delta_weight().shape
        ),
    )

    print("\nГлавный вывод:")
    print(
        "LoRA сохраняет Base weights замороженными "
        "и обучает небольшое low-rank обновление."
    )


if __name__ == "__main__":
    main()
