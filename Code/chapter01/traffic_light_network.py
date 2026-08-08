"""Первая учебная нейронная сеть проекта Almaz_AI.

Пример обучает небольшую модель принимать решение
«ИДТИ» или «СТОЯТЬ» по трём сигналам учебного светофора.

Этот файл является чистой Python-версией лабораторной работы:
Labs/chapter01/01_traffic_light_network.ipynb
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


RANDOM_SEED = 42
EPOCHS = 1000
LEARNING_RATE = 0.05
DECISION_THRESHOLD = 0.5


class TrafficLightNetwork(nn.Module):
    """Небольшая нейронная сеть для учебной задачи со светофором."""

    def __init__(self) -> None:
        """Создаёт два линейных слоя и функцию активации ReLU."""

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход и возвращает логиты модели."""

        return self.network(features)


def create_traffic_light_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Создаёт полный датасет состояний светофора и правильных решений."""

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
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        dtype=torch.float32,
    )

    return features, targets


def get_predictions(
    model: nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает вероятности движения и бинарные решения модели."""

    model.eval()

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= DECISION_THRESHOLD).int()

    return probabilities, predictions


def calculate_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Вычисляет долю правильных решений модели."""

    _, predictions = get_predictions(model, features)

    correct = predictions.float() == targets
    accuracy = correct.float().mean().item()

    return float(accuracy)


def train_model(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[list[float], list[float]]:
    """Обучает модель и возвращает историю Loss и Accuracy."""

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for epoch in range(epochs):
        model.train()

        # Forward Pass.
        logits = model(features)

        # Вычисляем величину ошибки.
        loss = loss_function(logits, targets)

        # Удаляем градиенты предыдущего шага.
        optimizer.zero_grad()

        # Backpropagation: вычисляем новые градиенты.
        loss.backward()

        # Optimizer изменяет веса и bias.
        optimizer.step()

        accuracy = calculate_accuracy(
            model=model,
            features=features,
            targets=targets,
        )

        loss_history.append(float(loss.item()))
        accuracy_history.append(accuracy)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Loss={loss.item():.6f} | "
                f"Accuracy={accuracy:.2%}"
            )

    return loss_history, accuracy_history


def create_prediction_table(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> pd.DataFrame:
    """Создаёт таблицу с состояниями светофора и ответами модели."""

    probabilities, predictions = get_predictions(
        model=model,
        features=features,
    )

    return pd.DataFrame(
        {
            "Красный": features[:, 0].int().numpy(),
            "Жёлтый": features[:, 1].int().numpy(),
            "Зелёный": features[:, 2].int().numpy(),
            "Вероятность ИДТИ": probabilities.squeeze().numpy(),
            "Ответ сети": [
                "ИДТИ" if value == 1 else "СТОЯТЬ"
                for value in predictions.squeeze().numpy()
            ],
            "Правильный ответ": [
                "ИДТИ" if value == 1 else "СТОЯТЬ"
                for value in targets.squeeze().int().numpy()
            ],
        }
    )


def predict_traffic_light(
    model: nn.Module,
    red: int,
    yellow: int,
    green: int,
) -> tuple[float, str]:
    """Возвращает вероятность движения и решение для одного состояния."""

    state = torch.tensor(
        [[float(red), float(yellow), float(green)]],
        dtype=torch.float32,
    )

    probabilities, predictions = get_predictions(
        model=model,
        features=state,
    )

    probability = float(probabilities.item())
    prediction = int(predictions.item())

    decision = "ИДТИ" if prediction == 1 else "СТОЯТЬ"

    return probability, decision


def show_model_parameters(model: nn.Module) -> None:
    """Выводит обучаемые параметры модели в консоль."""

    for name, parameter in model.named_parameters():
        print(f"\n{name}")
        print(f"shape: {tuple(parameter.shape)}")
        print(parameter.detach())


def plot_loss_history(loss_history: list[float]) -> None:
    """Строит график изменения функции потерь по эпохам."""

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Обучение нейросети «Светофор»: Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def plot_accuracy_history(accuracy_history: list[float]) -> None:
    """Строит график изменения точности модели по эпохам."""

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_history)
    plt.title("Обучение нейросети «Светофор»: Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()


def main() -> None:
    """Запускает полный учебный цикл создания и обучения модели."""

    torch.manual_seed(RANDOM_SEED)

    features, targets = create_traffic_light_dataset()

    model = TrafficLightNetwork()

    print("=== МОДЕЛЬ ===")
    print(model)

    print("\n=== ПРЕДСКАЗАНИЯ ДО ОБУЧЕНИЯ ===")
    predictions_before = create_prediction_table(
        model=model,
        features=features,
        targets=targets,
    )
    print(predictions_before.to_string(index=False))

    print("\n=== ОБУЧЕНИЕ ===")
    loss_history, accuracy_history = train_model(
        model=model,
        features=features,
        targets=targets,
    )

    print("\n=== ПРЕДСКАЗАНИЯ ПОСЛЕ ОБУЧЕНИЯ ===")
    predictions_after = create_prediction_table(
        model=model,
        features=features,
        targets=targets,
    )
    print(predictions_after.to_string(index=False))

    final_accuracy = calculate_accuracy(
        model=model,
        features=features,
        targets=targets,
    )

    print(f"\nИтоговая точность: {final_accuracy:.2%}")

    print("\n=== ОБУЧЕННЫЕ ПАРАМЕТРЫ ===")
    show_model_parameters(model)

    probability, decision = predict_traffic_light(
        model=model,
        red=0,
        yellow=1,
        green=1,
    )

    print("\n=== РУЧНАЯ ПРОВЕРКА [0, 1, 1] ===")
    print(f"Вероятность ИДТИ: {probability:.4f}")
    print(f"Решение: {decision}")

    plot_loss_history(loss_history)
    plot_accuracy_history(accuracy_history)


if __name__ == "__main__":
    main()
