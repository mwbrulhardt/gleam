#!/usr/bin/env bash
set -e

# Install act
gh extension install https://github.com/nektos/gh-act

sudo chown vscode .venv || true

# make the python binary location predictable
poetry config virtualenvs.in-project true


poetry install --with dev,test,typing --all-extras