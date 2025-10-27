# app/core/data_ingest.py
import os
from app.core.azure_blob import AzureBlobHandler
from app.core.text_extractor import TextExtractor
from app.core.chunker import TextChunker
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DataIngestor:
    """Handles document upload, extraction, and chunking."""
    
    def __init__(self):
        self.blob_handler = AzureBlobHandler()
        self.extractor = TextExtractor()
        self.chunker = TextChunker()

    def process_and_upload(self, local_file_path: str, blob_name: str):
        """Uploads a document and returns text chunks."""
        try:
            logger.info(f"📄 Processing file: {local_file_path}")
            
            # Upload to Azure
            self.blob_handler.upload_file(local_file_path, blob_name)

            # Extract text
            text = self.extractor.extract_text(local_file_path)

            # Split text into chunks
            chunks = self.chunker.chunk_text(text)
            logger.info(f"✅ Successfully created {len(chunks)} text chunks.")

            return chunks
        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
            raise e
