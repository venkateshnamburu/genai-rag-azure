# app/core/local_embedding.py
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class LocalEmbedding:
    """Wrapper for SentenceTransformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded embedding model: {model_name}")
        except Exception as e:
            logger.exception(f"❌ Failed to load embedding model {model_name}: {e}")
            raise

    def embed(self, texts):
        """Return list of embeddings for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
