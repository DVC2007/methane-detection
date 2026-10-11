# Contributing to the methane-detection project

## Branch policy

- `main` is the shared stable branch.
- `leader/integration` is the team-leader integration branch.
- Each member works on a short-lived branch created from an up-to-date base branch.
- Do not force-push shared branches.
- Do not commit raw data, secrets, access tokens, or very large model files.

## Start a task

```bash
cd /Users/mac4dvc/Documents/MDMA/methane-detection
git checkout main
git pull origin main
git checkout -b member1/data-preprocessing
```

Replace the branch name with your own task name. Use one branch per deliverable.

## Before opening a pull request

```bash
git status
pytest -q
git diff --check
git add .
git commit -m "Add <short description>"
git push -u origin member1/data-preprocessing
```

Complete the pull-request template. Include what changed, how it was tested, what remains uncertain, and what another member should review.

## Review rules

Before merging, the team leader confirms that the pull request targets the correct branch, is up to date, passes the `test` GitHub Actions check, has at least one teammate review, contains no secrets or unapproved data, and updates its associated task.

## Safe Codex use

Codex may inspect files, explain errors, propose code, write tests, and prepare documentation. It must not invent dataset facts or experimental results. The team verifies generated code, data licences, metrics, citations, and scientific claims. Record substantial AI assistance in `reports/ai_usage.md`.
