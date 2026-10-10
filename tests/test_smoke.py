
from pathlib import Path

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


def test_latest_run_artifacts_exist() -> None:
    results_dir = REPOSITORY_ROOT / "results"

    run_dirs = sorted(
        path
        for path in results_dir.glob("toy_*")
        if path.is_dir()
    )

    assert run_dirs, "No toy training run found. Run the toy training script first."

    latest_run = run_dirs[-1]

    for filename in ("config.yaml", "seed.txt", "checkpoint.pt"):
        assert (latest_run / filename).is_file(), (
            f"Missing {filename} in {latest_run}"
        )

    saved_config = yaml.safe_load(
        (latest_run / "config.yaml").read_text(encoding="utf-8")
    )
    saved_seed = int(
        (latest_run / "seed.txt").read_text(encoding="utf-8").strip()
    )

    assert saved_config["seed"] == saved_seed
