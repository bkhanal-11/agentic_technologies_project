import os

# Environment configuration
CONFIG = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "timeout": 60,  # seconds
    "max_results": 20,
    "relevance_threshold": 0.7  # Threshold for determining relevant papers (0-1)
}