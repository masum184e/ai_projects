from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

@dataclass
class Config:
    data_dir: str = "docs"
    index_dir: str = "index"

    chunk_size: int = 600
    chunk_overlap: int = 100

    embedding_model: SentenceTransformer = field(
        default_factory=lambda: SentenceTransformer("all-MiniLM-L6-v2")
    )