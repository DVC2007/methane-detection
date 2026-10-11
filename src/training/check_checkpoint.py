
import torch

from src.models.toy_segmentation import ToySegmentationModel


def main():
    checkpoint_path = (
        "results/toy_20261010_121111/checkpoint.pt"
    )

    # Load the saved checkpoint on the CPU.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Create a fresh model and load the saved weights.
    model = ToySegmentationModel(in_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test the model on four fake images.
    images = torch.rand(4, 3, 32, 32)

    with torch.no_grad():
        predictions = model(images)

    print("Checkpoint loaded successfully!")
    print("Saved epoch:", checkpoint["epoch"])
    print("Saved seed:", checkpoint["seed"])
    print("Input shape:", images.shape)
    print("Output shape:", predictions.shape)


if __name__ == "__main__":
    main()
