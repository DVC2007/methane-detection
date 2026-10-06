from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_project_scaffold_exists() -> None:
    expected_paths = [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "configs" / "base.yaml",
        REPOSITORY_ROOT / "data" / "README.md",
        REPOSITORY_ROOT / "reports" / "decisions.md",
    ]
    missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in expected_paths if not path.is_file()]
    assert not missing, f"Missing project scaffold files: {missing}"


def test_base_config_contains_reproducibility_fields() -> None:
    config_text = (REPOSITORY_ROOT / "configs" / "base.yaml").read_text()
    for required_field in ("seed:", "manifest:", "split_file:", "primary_metric:"):
        assert required_field in config_text
