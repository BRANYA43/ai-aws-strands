from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SESSION_STORAGE_DIR = BASE_DIR / 'tmp/sessions'

API_KEY = 'API_KEY'

MODEL_ID = 'gemini/gemini-3-flash-preview'

EXA_API_KEY = 'EXA_API_KEY'

LANGFUSE_SECRET_KEY = 'LANGFUSE_SECRET_KEY'
LANGFUSE_PUBLIC_KEY = 'LANGFUSE_PUBLIC_KEY'
LANGFUSE_BASE_URL = 'https://cloud.langfuse.com'