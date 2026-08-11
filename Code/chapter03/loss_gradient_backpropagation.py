"""Глава 3: Loss, Gradient Descent и Backpropagation.

Чистый Python-пример проекта Almaz_AI.

Скрипт показывает:
- MSE;
- BCEWithLogitsLoss;
- gradients;
- один шаг SGD;
- полный training loop;
- влияние Learning Rate.
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


RANDOM_SEED = 42
EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.05
DECISION_THRESHOLD = 0.5


class TrafficLightNetwork(nn.Module):
    """Маленькая нейросеть для учебной задачи светофора."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Возвращает logits модели."""
        return self.network(features)


def create_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Создаёт полный датасет состояний учебного светофора."""

    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0]],
        dtype=torch.float32,
    )

    return features, targets


def demonstrate_mse() -> None:
    """Показывает простой расчёт MSE."""

    prediction = torch.tensor(0.7)
    target = torch.tensor(1.0)
    manual_loss = (prediction - target) ** 2

    print("=== MSE ===")
    print("Prediction:", prediction.item())
    print("Target:", target.item())
    print("Manual MSE:", manual_loss.item())


def demonstrate_bce_with_logits() -> None:
    """Показывает связь logits, Sigmoid и BCEWithLogitsLoss."""

    logit = torch.tensor([[1.5]])
    target = torch.tensor([[1.0]])
    loss_function = nn.BCEWithLogitsLoss()

    loss = loss_function(logit, target)
    probability = torch.sigmoid(logit)

    print("\n=== BCEWithLogitsLoss ===")
    print("Logit:", logit.item())
    print("Probability:", probability.item())
    print("Loss:", loss.item())


def inspect_one_training_step(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Показывает один шаг SGD и gradient конкретного веса."""

    torch.manual_seed(RANDOM_SEED)

    model = TrafficLightNetwork()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    first_layer = model.network[0]
    weight_before = first_layer.weight.data[0, 0].item()

    logits = model(features)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()

    gradient = first_layer.weight.grad[0, 0].item()

    optimizer.step()

    weight_after = first_layer.weight.data[0, 0].item()
    expected_weight = weight_before - 0.1 * gradient

    print("\n=== ОДИН ШАГ ОБУЧЕНИЯ ===")
    print("Loss:", loss.item())
    print("Вес ДО:", weight_before)
    print("Gradient:", gradient)
    print("Вес ПОСЛЕ:", weight_after)
    print("Ожидаемый вес по формуле SGD:", expected_weight)


def calculate_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Вычисляет бинарную Accuracy."""

    model.eval()

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= DECISION_THRESHOLD).float()

    return float(
        (predictions == targets)
        .float()
        .mean()
        .item()
    )


def train_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = EPOCHS,
) -> tuple[TrafficLightNetwork, list[float], float]:
    """Обучает модель и возвращает модель, историю Loss и Accuracy."""

    torch.manual_seed(RANDOM_SEED)

    model = TrafficLightNetwork()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_history: list[float] = []

    for _ in range(epochs):
        model.train()

        logits = model(features)
        loss = loss_function(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))

    accuracy = calculate_accuracy(
        model=model,
        features=features,
        targets=targets,
    )

    return model, loss_history, accuracy


def compare_learning_rates(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[pd.DataFrame, dict[float, list[float]]]:
    """Сравнивает несколько значений Learning Rate."""

    learning_rates = [0.001, 0.01, 0.05, 0.5]
    histories: dict[float, list[float]] = {}
    rows: list[dict[str, float]] = []

    for learning_rate in learning_rates:
        _, history, accuracy = train_model(
            features=features,
            targets=targets,
            learning_rate=learning_rate,
        )

        histories[learning_rate] = history
        rows.append(
            {
                "Learning Rate": learning_rate,
                "Final Loss": history[-1],
                "Accuracy": accuracy,
            }
        )

    return pd.DataFrame(rows), histories


def plot_loss(
    history: list[float],
    title: str,
) -> None:
    """Строит график Loss."""

    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def plot_learning_rate_comparison(
    histories: dict[float, list[float]],
) -> None:
    """Строит графики Loss для разных Learning Rate."""

    plt.figure(figsize=(11, 6))

    for learning_rate, history in histories.items():
        plt.plot(
            history,
            label=f"lr={learning_rate}",
        )

    plt.title("Влияние Learning Rate на Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """Запускает демонстрации главы 3."""

    demonstrate_mse()
    demonstrate_bce_with_logits()

    features, targets = create_dataset()

    inspect_one_training_step(
        features=features,
        targets=targets,
    )

    _, history, accuracy = train_model(
        features=features,
        targets=targets,
    )

    print("\n=== ПОЛНОЕ ОБУЧЕНИЕ ===")
    print("Final Loss:", history[-1])
    print("Accuracy:", accuracy)

    plot_loss(
        history=history,
        title="Loss во время обучения",
    )

    results, histories = compare_learning_rates(
        features=features,
        targets=targets,
    )

    print("\n=== LEARNING RATE ===")
    print(results.round(6).to_string(index=False))

    plot_learning_rate_comparison(histories)

    print("\nГлавная цепочка:")
    print(
        "Forward Pass → Loss → zero_grad() → "
        "backward() → gradients → step() → updated parameters"
    )


if __name__ == "__main__":
    main()
