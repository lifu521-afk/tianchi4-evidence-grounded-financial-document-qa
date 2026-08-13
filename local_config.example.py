"""Local-only LLM settings.

Copy this file to ``local_config.py`` and fill in the credentials locally.
Never commit the copied file or an API key.
"""

PROVIDER = "qwen"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = ""
MODEL = "qwen3.6-plus"

TEMPERATURE = 0
TIMEOUT_SECONDS = 600
MAX_RETRIES = 5
ALLOW_NON_QWEN = False

# Default VS Code / train.py behavior.
RUN_MODE = "targeted"
OUTPUT_DIR = "runs"
SYNC_TO_SUBMISSION = False
SUBMISSION_ANSWER_CSV = "answer.csv"

EVIDENCE_MODE = "compact"
REVIEW_MODE = "auto"
REVIEW_POLICY = "evidence_gate"
MAX_CONTEXT_CHARS = 0
LIMIT = None
PREPROCESS_BEFORE_RUN = False
