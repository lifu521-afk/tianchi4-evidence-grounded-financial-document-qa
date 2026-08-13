# Security Notes

This project calls OpenAI-compatible Qwen endpoints. Credentials are loaded
from environment variables or the local-only `local_config.py` file.

- Copy `local_config.example.py` to `local_config.py` for local use.
- Do not commit `local_config.py`, `.env`, API keys, or generated evidence.
- If a key is exposed in a terminal, chat, log, or commit, revoke it at the
  provider immediately and create a replacement.
- Use a relay endpoint only when it is authorized by the competition rules.
- Keep competition datasets and answer files outside the public repository.
