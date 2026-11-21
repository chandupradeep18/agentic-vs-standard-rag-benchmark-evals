from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        self.splitter=RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""])

    def chuck_documents(self,documents: List[Any])->list[Any]:
        if not documents:
            return []
        chunks=self.splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def convert_chunks_to_embedding(self,chunks:list[Any])->np.ndarray:
        if not chunks:
            return np.array([])
        texts=[chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def convert_chunks_to_embedding_query(self,chunks:list[Any])->np.ndarray:
        if not chunks:
            return np.array([])
        embeddings=self.model.encode(chunks,show_progress_bar=False)
        return embeddings
