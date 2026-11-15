import numpy as np
from dataclasses import dataclass, field
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.array(vector, dtype=np.float32)
        
@dataclass
class Config:
    data_dir: str = "docs"

    chunk_size: int = 600
    chunk_overlap: int = 100

    embedding_model_name: str = "all-MiniLM-L6-v2"

    store_type: str = "faiss"
    index_dir: str = "index"
    collection_name: str = "rag_documents"