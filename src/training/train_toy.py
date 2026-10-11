
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.synthetic import SyntheticMethaneDataset
from src.models.toy_segmentation import ToySegmentationModel
from src.utils.config import load_config
from src.utils.reproducibility import get_device, set_seed


def main():
    # 1. Load settings from the YAML configuration.
    config = load_config()

    seed = config["seed"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    # 2. Create a unique folder for this training run.
    run_id = datetime.now().strftime("toy_%Y%m%d_%H%M%S")
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # 3. Save the configuration and random seed.
    shutil.copy2("configs/base.yaml", run_dir / "config.yaml")

    (run_dir / "seed.txt").write_text(
        f"{seed}\n",
        encoding="utf-8",
    )

    # 4. Set the seed and select the computing device.
    set_seed(seed)

    configured_device = config["training"]["device"]

    if configured_device == "auto":
        device = get_device()
    else:
        device = torch.device(configured_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    # 5. Create the synthetic dataset.
    dataset = SyntheticMethaneDataset(
        num_samples=100,
        image_size=32,
        num_channels=3,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # 6. Create the CNN, loss function, and optimizer.
    model = ToySegmentationModel(in_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print(f"Run ID: {run_id}")
    print(f"Run folder: {run_dir}")
    print(f"Device: {device}")
    print(f"Training samples: {len(dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # 7. Train the model.
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Predict scores for every pixel.
            predictions = model(images)

            # Compare predictions with the correct masks.
            loss = criterion(predictions, masks)

            # Calculate gradients and update model weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: {average_loss:.4f}"
        )

    # 8. Save the trained model and training information.
    checkpoint_path = run_dir / "checkpoint.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "seed": seed,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training finished. Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
