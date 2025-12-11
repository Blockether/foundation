# Developing Blockether Foundation

## Workflow

### Start with an issue before writing code

Before writing any code, please create an issue first that describes the problem
you are trying to solve with alternatives that you have considered. A little bit
of prior communication can save a lot of time on coding. Keep the problem as
small as possible. If there are two problems, make two issues. We discuss the
issue and if we reach an agreement on the approach, it's time to move on to a
PR.

### Follow up with a pull request

Post a corresponding PR with the smallest change possible to address the
issue. Then we discuss the PR, make changes as needed and if we reach an
agreement, the PR will be merged.

### Tests

Each bug fix, change or new feature should be tested well to prevent future
regressions.

If possible, tests should use public APIs. If the bug is in private/internal
code, try to trigger it from a public API.

### Force-push

Please do not use `git push --force` on your PR branch for the following
reasons:

- It makes it more difficult for others to contribute to your branch if needed.
- It makes it harder to review incremental commits.
- Links (in e.g. e-mails and notifications) go stale and you're confronted with:
  this code isn't here anymore, when clicking on them.
- GitHub Actions doesn't play well with it: it might try to fetch a commit which
  doesn't exist anymore.
- Your PR will be squashed anyway.

## Requirements

You need [Python 3.13+](https://www.python.org/downloads/) for development and [uv](https://docs.astral.sh/uv/) for package management. For testing and type checking you'll need the development dependencies.

You need `poe` installed globally via brew. Use `brew tap nat-n/poethepoet && brew install nat-n/poethepoet/poethepoet`

## Clone repository

```bash
git clone https://github.com/blockether/blockether-foundation.git
cd blockether-foundation
```

## Development Setup

Set up your development environment:

```bash
# Create virtual environment
uv venv

# Activate (Unix/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev]"
```

### Langwatch Setup

```
cd ../ && git clone https://github.com/langwatch/langwatch.git
cd langwatch
cp langwatch/.env.example langwatch/.env
docker compose up -d --wait --build
open http://localhost:5560
```