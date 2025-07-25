import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "jina_api_key": os.getenv("JINA_API_KEY"),
    "timeout": 60,
    "max_results": 20,
    "relevance_threshold": 0.7
}