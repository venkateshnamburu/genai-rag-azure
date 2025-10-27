from azure.storage.blob import BlobServiceClient
from app.core.config import settings
from app.utils.logger import get_logger
from datetime import datetime
import json
import os

logger = get_logger(__name__)

class AzureBlobHandler:
    def __init__(self):
        try:
            self.connection_string = settings.AZURE_CONNECTION_STRING
            self.container_name = settings.AZURE_CONTAINER_NAME
            self.service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.service_client.get_container_client(self.container_name)
            logger.info("✅ Azure Blob Storage client initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Blob Storage: {e}")
            raise e

    def upload_file(self, file_path: str, blob_name: str):
        """Upload a file to Azure Blob Storage."""
        try:
            with open(file_path, "rb") as data:
                self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            logger.info(f"✅ Uploaded file to blob: {blob_name}")
        except Exception as e:
            logger.error(f"❌ Error uploading file to Azure Blob Storage: {e}")
            raise e

    def upload_text(self, username: str, question: str, response: str):
        """Upload chat interaction (username, question, response) to Azure Blob Storage as JSON."""
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            blob_name = f"chat_logs/{username}_{timestamp}.json"

            chat_record = {
                "username": username,
                "timestamp": timestamp,
                "question": question,
                "response": response
            }

            json_data = json.dumps(chat_record, indent=4)
            self.container_client.upload_blob(name=blob_name, data=json_data, overwrite=True)
            logger.info(f"✅ Uploaded chat log for user '{username}' as {blob_name}")
        except Exception as e:
            logger.error(f"❌ Failed to upload chat log: {e}")
            raise e

    def list_files(self, container_name: str = None):
        """List all blobs in a container (uses default container if not specified)."""
        try:
            container = container_name or self.container_name
            container_client = self.service_client.get_container_client(container)
            blobs = [blob.name for blob in container_client.list_blobs()]
            logger.info(f"📄 Found {len(blobs)} files in container '{container}'")
            return blobs
        except Exception as e:
            logger.error(f"❌ Error listing blobs: {e}")
            return []

    def download_file(self, container_name: str, blob_name: str, download_path: str):
        """Download a blob to a local file."""
        try:
            container_client = self.service_client.get_container_client(container_name)
            with open(download_path, "wb") as file:
                data = container_client.download_blob(blob_name)
                file.write(data.readall())
            logger.info(f"✅ Downloaded blob: {blob_name} to {download_path}")
        except Exception as e:
            logger.error(f"❌ Error downloading blob: {e}")
            raise e
