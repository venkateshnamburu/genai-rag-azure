from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Splits text into overlapping chunks optimized for document retrieval (RAG).
    Keeps related sentences like equations, definitions, or lists in the same chunk.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Args:
            chunk_size (int): Number of characters per chunk (default: 800)
            chunk_overlap (int): Overlap between chunks (default: 150)
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # paragraph
                "\n",    # line
                ".",     # sentence
                "!", 
                "?",
                ";",
                " ", 
                ""       # fallback
            ]
        )

    def chunk_text(self, text: str):
        """Split a single document into cleaned, non-empty chunks."""
        if not text or not isinstance(text, str):
            return []

        chunks = self.splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
