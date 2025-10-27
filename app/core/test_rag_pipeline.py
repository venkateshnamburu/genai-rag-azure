import logging
from app.core.rag_engine import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🧠 Starting RAG pipeline test...")

    rag = RAGPipeline()

    # Step 1: Process PDFs from Azure Blob
    container_name = rag.blob.container_name
    logger.info(f"📦 Processing container: {container_name}")
    rag.process_blobs(container_name)

    # Step 2: Test query
    query = "Comparison of Different Rechargeable Batteries with their Energy Density?"
    logger.info(f"💬 Querying: {query}")

    response = rag.query(query)

    print("\n" + "="*80)
    print("🤖 RAG Response (JSON):\n")
    print(response)
    print("="*80)
