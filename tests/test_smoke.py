
from pathlib import Path
import shutil

import torch
import yaml

from src.data.synthetic import SyntheticMethaneDataset
from src.models.toy_segmentation import ToySegmentationModel
from src.utils.config import load_config
from src.utils.reproducibility import get_device, set_seed


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_project_scaffold_exists() -> None:
    expected_paths = [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "configs" / "base.yaml",
        REPOSITORY_ROOT / "data" / "README.md",
        REPOSITORY_ROOT / "reports" / "decisions.md",
    ]

    missing = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in expected_paths
        if not path.is_file()
    ]

    assert not missing, f"Missing project scaffold files: {missing}"


def test_base_config_contains_reproducibility_fields() -> None:
    config = load_config(REPOSITORY_ROOT / "configs" / "base.yaml")

    assert "seed" in config
    assert "manifest" in config["data"]
    assert "split_file" in config["data"]
    assert "primary_metric" in config["evaluation"]


def test_reproducibility_and_device() -> None:
    set_seed(42)
    first = torch.rand(3)

    set_seed(42)
    second = torch.rand(3)

    assert torch.equal(first, second)
    assert get_device().type in ("cpu", "cuda")


def test_synthetic_dataset_shapes() -> None:
    dataset = SyntheticMethaneDataset(
        num_samples=4,
        image_size=32,
        num_channels=3,
        seed=42,
    )

    image, mask = dataset[0]

    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)


def test_model_output_shape() -> None:
    model = ToySegmentationModel(in_channels=3)
    images = torch.rand(4, 3, 32, 32)

    predictions = model(images)

    assert predictions.shape == (4, 1, 32, 32)


def test_checkpoint_save_and_reload(tmp_path: Path) -> None:
    model = ToySegmentationModel(in_channels=3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "seed": 42,
        },
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    restored_model = ToySegmentationModel(in_channels=3)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_model.eval()

    with torch.no_grad():
        predictions = restored_model(torch.rand(4, 3, 32, 32))

    assert checkpoint["epoch"] == 10
    assert checkpoint["seed"] == 42
    assert predictions.shape == (4, 1, 32, 32)


def test_training_artifacts_are_created(tmp_path: Path) -> None:
    """Verify that the required training artifacts can be created."""
    config_path = REPOSITORY_ROOT / "configs" / "base.yaml"
    run_dir = tmp_path / "toy_test_run"
    run_dir.mkdir()

    shutil.copy2(config_path, run_dir / "config.yaml")

    (run_dir / "seed.txt").write_text(
        "42\n",
        encoding="utf-8",
    )

    model = ToySegmentationModel(in_channels=3)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "seed": 42,
        },
        run_dir / "checkpoint.pt",
    )

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "seed.txt").is_file()
    assert (run_dir / "checkpoint.pt").is_file()

    saved_config = yaml.safe_load(
        (run_dir / "config.yaml").read_text(encoding="utf-8")
    )

    saved_seed = int(
        (run_dir / "seed.txt").read_text(encoding="utf-8").strip()
    )

    assert saved_config["seed"] == saved_seed
