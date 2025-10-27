import os
import re
import json
import logging
from app.core.azure_blob import AzureBlobHandler
from app.core.text_extractor import TextExtractor
from app.core.chunker import TextChunker
from app.core.local_embedding import LocalEmbedding
from app.core.pinecone_client import PineconeClient
from app.core.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline: extract → embed → store → retrieve → generate (JSON output)"""

    def __init__(self):
        logger.info("🚀 Initializing Enhanced RAG Pipeline with JSON response format...")
        self.blob = AzureBlobHandler()
        self.extractor = TextExtractor()
        self.chunker = TextChunker(chunk_size=800, chunk_overlap=150)
        self.embedder = LocalEmbedding(model_name="all-mpnet-base-v2")
        self.vectorstore = PineconeClient()
        self.llm = GeminiClient()

    def process_blobs(self, container_name: str):
        """Load, chunk, and store embeddings from Azure PDFs."""
        files = self.blob.list_files(container_name)
        logger.info(f"📦 Found {len(files)} files in Azure Blob container '{container_name}'")

        for file in files:
            # ✅ Skip non-PDF files (e.g., chat logs or metadata)
            if not file.lower().endswith(".pdf"):
                logger.info(f"⏭️ Skipping non-PDF file: {file}")
                continue

            safe_filename = os.path.basename(file)
            download_path = f"temp_{safe_filename}"
            logger.info(f"📄 Processing file: {file}")

            try:
                self.blob.download_file(container_name, file, download_path)
                text = self.extractor.extract_text(download_path)
                chunks = self.chunker.chunk_text(text)
                logger.info(f"✅ Created {len(chunks)} chunks from {file}")

                embeddings = self.embedder.embed(chunks)
                self.vectorstore.upsert_embeddings(chunks, embeddings, file_name=file)
                os.remove(download_path)
                logger.info(f"🧠 Embedded and stored vectors for: {file}")

            except Exception as e:
                logger.error(f"❌ Error processing file {file}: {e}")
                if os.path.exists(download_path):
                    os.remove(download_path)
                continue

    def query(self, query: str, top_k: int = 5):
        """Query the vector store and generate structured JSON response."""
        logger.info(f"💬 Query received: {query}")
        query_vector = self.embedder.embed([query])[0]
        results = self.vectorstore.query(query_vector, top_k=top_k)

        if not results:
            return {
                "answer": "No relevant information found in the provided context.",
                "relevant_documents": []
            }

        # Build context string for LLM
        context = ""
        for match in results:
            filename = match["metadata"].get("source", "unknown")
            chunk_text = match["metadata"].get("text", "")
            context += f"[Document: {filename}]\n{chunk_text}\n\n"

        prompt = f"""
You are a highly accurate AI assistant that answers questions ONLY using the provided context.

If the context does not contain the information, say exactly:
"No relevant information found in the provided context."

You must respond in valid JSON only, with this structure:
{{
  "answer": "string",
  "relevant_documents": [
    {{
      "filename": "string",
      "matched_chunks": ["string", ...]
    }}
  ]
}}

Use only the facts from the context. Do not make assumptions or add outside knowledge.

Context:
{context}

Question:
{query}
"""

        raw_response = self.llm.generate_response(prompt)
        logger.info("✅ Raw LLM response received.")

        # ---- 🧹 Clean and parse JSON safely ----
        cleaned = re.sub(r"```json|```", "", raw_response).strip()
        try:
            parsed = json.loads(cleaned)
            logger.info("✅ Parsed LLM response as valid JSON.")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error: {e}. Returning raw text instead.")
            return {"answer": cleaned, "relevant_documents": []}
