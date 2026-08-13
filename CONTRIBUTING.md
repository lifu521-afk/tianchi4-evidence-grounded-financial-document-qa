# Contributing

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item local_config.example.py local_config.py
```

Fill in `API_KEY` in the local copy, or configure the supported environment
variables. Run retrieval and format checks before calling the model:

```powershell
python script\inspect_dataset.py
python train.py --mode preprocess
python train.py --mode dry-run
python -m pytest
```

Generated data, caches, model outputs, and submission files are intentionally
ignored by Git.
