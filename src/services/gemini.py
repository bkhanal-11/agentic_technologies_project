import aiohttp

from utils.logger import logger

class GeminiLLMService:
    """Service to interact with Google Gemini API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
    async def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini Pro model"""
        params = {
            "key": self.api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                params=params,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract text from response
                    content = result.get("candidates", [{}])[0].get("content", {})
                    parts = content.get("parts", [{}])
                    return parts[0].get("text", "No response generated")
                else:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: {error_text}")
                    return f"Error: {response.status} - {error_text}"