# AI Coding Session Notes

Use this file as the project handoff context for AI coding tools.

## Required Python Interpreter

- Always use: `~/miniconda3/envs/py312/bin/python`
- VS Code workspace default is pinned in `.vscode/settings.json`.

## Environment Setup
for Ampere GPU,
```bash
~/miniconda3/envs/py312/bin/python -m pip install -r requirements.txt
```
for Turing GPU,
```bash
~/miniconda3/envs/py312/bin/python -m pip install -r requirements_pre_ampere.txt
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

- Main flash-attention implementation: `flash_atten.py`
- Shared tuner abstractions: `tuner.py`
- Resource error/report formatting helpers: `exceptions.py`
- General correctness/performance tests: `test_flash_atten.py`
- Profiling-focused tests: `test_profile.py`
- Dependencies: `requirements.txt`
- CUDA PyTorch wheel index is configured in `requirements.txt` (`cu128`).

## Current Behavior Notes

- - `flash_attention(...)` in the flash_atten.py is the unified entrance for all flash attention kernel functions.
- Flash-attention launch configuration selection is handled by `FlashAttentionTuner`; callers and tests should not precompute `BLOCK_M`.