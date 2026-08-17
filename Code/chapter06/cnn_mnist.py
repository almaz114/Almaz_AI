"""Глава 6: CNN — свёрточные нейронные сети.

Чистый Python-пример проекта Almaz_AI.

Скрипт:
- загружает MNIST;
- создаёт CNN;
- обучает её;
- оценивает Accuracy;
- визуализирует Feature Maps;
- сохраняет state_dict модели.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001
DATA_DIR = Path("Datasets/mnist")
MODEL_PATH = Path("mnist_cnn.pth")


class MNISTCNN(nn.Module):
    """Простая CNN для классификации MNIST."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Возвращает logits для десяти классов."""
        features = self.features(images)
        return self.classifier(features)


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Создаёт Train и Test DataLoader."""

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Обучает модель одну эпоху."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_function(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Оценивает модель без изменения параметров."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = loss_function(logits, targets)

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return total_loss / len(loader), correct / total


def train_model(
    model: MNISTCNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
) -> dict[str, list[float]]:
    """Обучает CNN и возвращает историю метрик."""

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_function,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    return history


def plot_history(
    history: dict[str, list[float]],
) -> None:
    """Строит графики Loss и Accuracy."""

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.title("CNN MNIST Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_accuracy"], label="Train")
    plt.plot(history["test_accuracy"], label="Test")
    plt.title("CNN MNIST Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_feature_maps(
    model: MNISTCNN,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """Визуализирует Feature Maps первого Conv-слоя."""

    model.eval()

    images, _ = next(iter(loader))
    image = images[:1].to(device)

    with torch.no_grad():
        conv_output = model.features[0](image)
        activated = model.features[1](conv_output)

    maps = activated[0].cpu()

    plt.figure(figsize=(12, 6))

    for index in range(min(8, maps.shape[0])):
        plt.subplot(2, 4, index + 1)
        plt.imshow(maps[index], cmap="gray")
        plt.title(f"Feature {index}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Запускает учебный pipeline CNN."""

    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, test_loader = create_dataloaders()

    model = MNISTCNN().to(device)

    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Всего параметров:", total_parameters)

    history = train_model(
        model,
        train_loader,
        test_loader,
        device,
    )

    plot_history(history)

    show_feature_maps(
        model,
        test_loader,
        device,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print("Модель сохранена:", MODEL_PATH)


if __name__ == "__main__":
    main()
