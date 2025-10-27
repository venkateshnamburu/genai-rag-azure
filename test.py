import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Get values from .env
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX")
region = os.getenv("PINECONE_REGION", "us-east-1")
cloud = os.getenv("PINECONE_CLOUD", "aws")

print("🔑 Using Pinecone API Key:", api_key[:8] + "..." if api_key else "❌ Missing key")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Ensure index exists
if index_name not in [index.name for index in pc.list_indexes()]:
    print(f"⚙️ Creating Pinecone index '{index_name}' ...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match your embedding size
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
else:
    print(f"✅ Index '{index_name}' already exists.")

# Connect to index
index = pc.Index(index_name)
print(f"✅ Connected to Pinecone index: {index_name}")

# ---- Test data upload ----
vectors = [
    ("vec1", [0.1] * 384, {"text": "Hello world"}),
    ("vec2", [0.2] * 384, {"text": "This is a Pinecone test"}),
]

index.upsert(vectors)
print("✅ Upserted 2 test vectors.")

# ---- Query test ----
query_vector = [0.1] * 384
results = index.query(vector=query_vector, top_k=2, include_metadata=True)

print("🔍 Query Results:")
for match in results["matches"]:
    print(f"  ➤ ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")
