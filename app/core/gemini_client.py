# app/core/gemini_client.py
import google.generativeai as genai
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    """Handles text embedding and generation using Gemini."""

    def __init__(self):
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.embed_model = settings.EMBEDDING_MODEL
            self.gen_model = settings.GENERATIVE_MODEL
            logger.info("✅ Gemini client initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise e

    def get_embedding(self, text: str):
        """Generate embeddings for text using Gemini."""
        try:
            response = genai.embed_content(model=self.embed_model, content=text)
            embedding = response["embedding"]
            return embedding
        except Exception as e:
            logger.error(f"❌ Failed to get embedding: {e}")
            raise e

    def generate_text(self, prompt: str):
        """Generate text response using Gemini."""
        try:
            model = genai.GenerativeModel(self.gen_model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"❌ Failed to generate text: {e}")
            raise e

    def generate_response(self, prompt: str):
        """Wrapper for compatibility with RAG pipeline."""
        return self.generate_text(prompt)
