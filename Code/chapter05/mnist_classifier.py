"""Глава 5: MNIST — первая модель изображений.

Чистый Python-пример проекта Almaz_AI.

Скрипт:
- загружает MNIST;
- создаёт DataLoader;
- обучает MLP-классификатор;
- оценивает Test Accuracy;
- показывает предсказания;
- сохраняет и загружает state_dict модели.
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
MODEL_PATH = Path("mnist_model.pth")


class MNISTNetwork(nn.Module):
    """Полносвязная нейросеть для классификации цифр MNIST."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Возвращает logits для 10 классов."""
        return self.network(images)


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Создаёт Train и Test DataLoader для MNIST."""

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


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

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Оценивает Loss и Accuracy без обучения."""

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

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


def train_model(
    model: MNISTNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, list[float]]:
    """Обучает модель и возвращает историю метрик."""

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            loader=test_loader,
            loss_function=loss_function,
            device=device,
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
    plt.title("MNIST Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_accuracy"], label="Train")
    plt.plot(history["test_accuracy"], label="Test")
    plt.title("MNIST Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    count: int = 12,
) -> None:
    """Показывает несколько предсказаний модели."""

    model.eval()

    images, targets = next(iter(loader))
    device_images = images.to(device)

    with torch.no_grad():
        logits = model(device_images)
        predictions = logits.argmax(dim=1).cpu()

    plt.figure(figsize=(12, 6))

    for index in range(min(count, len(images))):
        plt.subplot(3, 4, index + 1)
        plt.imshow(
            images[index].squeeze(),
            cmap="gray",
        )
        plt.title(
            f"T={targets[index].item()} / "
            f"P={predictions[index].item()}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_model(
    model: nn.Module,
    path: Path = MODEL_PATH,
) -> None:
    """Сохраняет state_dict модели."""

    torch.save(
        model.state_dict(),
        path,
    )

    print("Модель сохранена:", path)


def load_model(
    path: Path,
    device: torch.device,
) -> MNISTNetwork:
    """Создаёт модель и загружает state_dict."""

    model = MNISTNetwork().to(device)

    state_dict = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main() -> None:
    """Запускает полный учебный pipeline MNIST."""

    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_loader, test_loader = create_dataloaders()

    model = MNISTNetwork().to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(model)
    print("Всего параметров:", total_parameters)

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )

    plot_history(history)

    show_predictions(
        model=model,
        loader=test_loader,
        device=device,
    )

    save_model(model)

    loaded_model = load_model(
        path=MODEL_PATH,
        device=device,
    )

    loss_function = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model=loaded_model,
        loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print(
        "Loaded model | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
