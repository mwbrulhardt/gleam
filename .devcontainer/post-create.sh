#!/usr/bin/env bash
set -e

sudo chown vscode .venv || true

# make the python binary location predictable
poetry config virtualenvs.in-project true


poetry install --with dev,test --all-extras