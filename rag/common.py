from dataclasses import dataclass

@dataclass
class Config:
    data_dir: str = "docs"
    index_dir: str = "index"

    chunk_size: int = 600
    chunk_overlap: int = 100