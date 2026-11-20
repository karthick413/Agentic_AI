import os

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "")
PRICE_IN_PER_M = float(os.getenv("LLM_PRICE_IN_PER_M", "0"))
PRICE_OUT_PER_M = float(os.getenv("LLM_PRICE_OUT_PER_M", "0"))
MAX_TOKENS_PARAM = os.getenv("LLM_MAX_TOKENS_PARAM", "max_tokens")
