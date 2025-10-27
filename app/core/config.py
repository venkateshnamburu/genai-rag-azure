# app/core/config.py
import os
from dotenv import load_dotenv

# Automatically detect project root (works from anywhere)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load environment variables
load_dotenv(ENV_PATH)


class Settings:
    """Centralized configuration for environment variables."""

    # Azure Storage
    AZURE_CONNECTION_STRING: str = os.getenv("AZURE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME: str = os.getenv("AZURE_CONTAINER_NAME")

    # Google Gemini API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # Pinecone Vector DB
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX")

    # Model Configuration
    EMBEDDING_MODEL: str = "models/embedding-001"      # Recommended embedding model
    GENERATIVE_MODEL: str = "gemini-2.0-flash"         # ✅ Updated to latest Gemini 2.0 Flash


settings = Settings()
