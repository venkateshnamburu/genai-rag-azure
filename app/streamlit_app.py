import sys
import os
import streamlit as st
from datetime import datetime
import json

# ------------------ Path Fix ------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ------------------ Imports ------------------
from app.core.rag_engine import RAGPipeline
from app.core.azure_blob import AzureBlobHandler

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Advanced AI Chatbot", page_icon="🤖", layout="centered")
st.title("🔍 Document-Aware AI Chatbot 🤖")

# Initialize session state
if "username" not in st.session_state:
    st.session_state.username = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "response" not in st.session_state:
    st.session_state.response = {}

# Initialize handlers
rag = RAGPipeline()
blob = AzureBlobHandler()

# Step 1: Ask username
st.session_state.username = st.text_input(
    "👤 Enter your name:",
    st.session_state.username,
    placeholder="Type your name to begin..."
)

# Proceed if username entered
if st.session_state.username:
    st.session_state.question = st.text_area(
        "💭 Ask a question related to the documents:",
        st.session_state.question,
        placeholder="Example: What are the main functions of a Battery Management System?",
        height=150
    )

    # Enable the button only when a question is typed
    if st.session_state.question.strip():
        if st.button("🚀 Submit"):
            with st.spinner("🔎 Searching knowledge base and generating answer..."):
                # Generate answer
                response = rag.query(st.session_state.question)
                st.session_state.response = response

                # Display JSON response
                st.subheader("🧠 AI Response (JSON Format):")
                st.json(response)

                # Save chat log to Azure Blob
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    data_to_store = {
                        "username": st.session_state.username,
                        "question": st.session_state.question,
                        "response": response,
                        "timestamp": timestamp
                    }

                    blob_name = f"chat_logs/{st.session_state.username}_{timestamp}.json"
                    temp_file = "temp_chat_log.json"

                    with open(temp_file, "w") as f:
                        json.dump(data_to_store, f, indent=4)

                    blob.upload_file(temp_file, blob_name)
                    os.remove(temp_file)

                    st.success("✅ Chat successfully stored in Azure Blob Storage!")

                except Exception as e:
                    st.error(f"❌ Failed to upload chat to Azure Blob: {e}")

    else:
        st.info("✍️ Please type your question to enable the **Submit** button.")
else:
    st.warning("👋 Please enter your name to start chatting.")

# Footer
st.markdown("---")
st.caption("💡 Powered by Gemini + Pinecone + Azure Blob Storage (Free Tier)")
