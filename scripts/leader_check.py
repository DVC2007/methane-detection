"""Run the team-leader checks for the current project session."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "CONTRIBUTING.md",
    ".github/workflows/tests.yml",
    ".github/pull_request_template.md",
    "configs/base.yaml",
    "data/README.md",
    "reports/decisions.md",
    "reports/daily_handover.md",
    "tests/test_smoke.py",
)


def run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    print(f"Repository: {ROOT}")

    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        print("FAIL: missing required files:")
        print("  " + "\n  ".join(missing))
        return 1
    print(f"PASS: all {len(REQUIRED)} leadership files exist")

    code, branch = run("git", "branch", "--show-current")
    if code != 0:
        print("FAIL: could not read the current branch")
        return 1
    print(f"Branch: {branch}")
    if branch == "main":
        print("NOTE: main is suitable for inspection only; members must create task branches.")

    code, whitespace = run("git", "diff", "--check")
    if code != 0:
        print("FAIL: whitespace errors found")
        print(whitespace)
        return 1
    print("PASS: git diff --check")

    pytest = ROOT / ".venv" / "bin" / "pytest"
    command = str(pytest) if pytest.exists() else sys.executable + " -m pytest"
    if pytest.exists():
        result = subprocess.run([str(pytest), "-q"], cwd=ROOT, text=True)
    else:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True)
    if result.returncode != 0:
        print("FAIL: pytest did not pass")
        return result.returncode
    print("PASS: pytest")
    print("\nLeader checks complete. Next: distribute the four Oct 7 task issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
