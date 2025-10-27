# app/core/pinecone_client.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from app.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)
load_dotenv()

class PineconeClient:
    """Handles Pinecone vector DB operations."""

    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        cloud = os.getenv("PINECONE_CLOUD", "aws")

        if not api_key:
            raise ValueError("❌ PINECONE_API_KEY not found in .env")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # ✅ The embedding model all-mpnet-base-v2 has dimension = 768
        dimension = 768

        # Check if index exists; if not, create with correct dimension
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"⚙️ Creating Pinecone index: {self.index_name} (dim={dimension})")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )

        # Connect to existing index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"✅ Connected to Pinecone index: {self.index_name}")

    def upsert_embeddings(self, chunks, embeddings, file_name=None):
        """Upsert text chunks + embeddings into Pinecone."""
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Ensure numpy array format
            emb_array = np.array(emb, dtype=float)

            vectors.append({
                "id": f"{file_name or 'chunk'}-{i}",
                "values": emb_array.tolist(),
                "metadata": {
                    "text": chunk,
                    "source": file_name or "unknown"
                }
            })

        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"✅ Upserted {len(vectors)} vectors from {file_name} into Pinecone.")
        else:
            logger.warning("⚠️ No vectors to upsert (empty chunks or embeddings).")

    def query(self, query_vector, top_k=3):
        """Query top-k similar vectors."""
        # Ensure query vector is numpy array
        query_array = np.array(query_vector, dtype=float)

        results = self.index.query(
            vector=query_array.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])
        logger.info(f"🔍 Retrieved {len(matches)} matches from Pinecone.")
        return matches
