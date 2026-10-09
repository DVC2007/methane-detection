#!/bin/zsh
set -e

repo="DVC2007/methane-detection"
root="$(cd "$(dirname "$0")/.." && pwd)"

gh auth status >/dev/null

for task in "$root"/tasks/oct7/*.md; do
  title="$(sed -n '1p' "$task" | sed 's/^# //')"
  echo "Creating: $title"
  gh issue create --repo "$repo" --title "$title" --body-file "$task"
done

echo "All October 7 task issues created."
