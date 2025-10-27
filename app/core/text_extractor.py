# app/core/text_extractor.py
import os
from PyPDF2 import PdfReader

class TextExtractor:
    """Extract text content from PDF or TXT files."""

    @staticmethod
    def extract_text(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            raise ValueError("Unsupported file type. Only .pdf and .txt are supported.")
