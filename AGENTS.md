# AI Coding Session Notes

Use this file as the project handoff context for AI coding tools.

## Required Python Interpreter

- Always use: `~/miniconda3/envs/py312/bin/python`
- VS Code workspace default is pinned in `.vscode/settings.json`.

## Environment Setup

```bash
~/miniconda3/envs/py312/bin/python -m pip install -r requirements.txt
```

## Test Commands

Run the main test script directly:

```bash
~/miniconda3/envs/py312/bin/python test_flash_atten.py
```

Run all pytest tests:

```bash
~/miniconda3/envs/py312/bin/python -m pytest -q
```

## Useful Project Context

- Main implementation: `flash_atten.py`
- Test script: `test_flash_atten.py`
- Dependencies: `requirements.txt`
- CUDA PyTorch wheel index is configured in `requirements.txt` (`cu128`).