import re
import time
import argparse
from pathlib import Path
from pypdf import PdfReader
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import Config

class DocumentLoader:
    """Reusable document loader for RAG applications."""

    SUPPORTED = {".txt", ".md"}
    # SUPPORTED = {".txt", ".md", ".pdf"}

    def __init__(self):
        self.skipped: list[str] = []

    def _load_file(self, path: Path) -> list[Document]:
        """Load an individual TXT, Markdown, or PDF file."""

        extension = path.suffix.lower()

        if extension == ".pdf":
            reader = PdfReader(str(path))
            documents = []

            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "page": page_number,
                        },
                    )
                )

            return documents

        # Load TXT and Markdown files
        text = path.read_text(encoding="utf-8-sig")

        if extension == ".md":
            text = re.sub(
                r"^#{1,6}\s*(.+)$",
                r"\1.",
                text,
                flags=re.MULTILINE,
            )

        return [
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        ]

    def load_dir(
        self,
        data_dir: str | Path,
    ) -> list[Document]:
        """
        Recursively load documents from a directory.

        Returns:
            List of LangChain Document objects.
        """

        root = Path(data_dir)

        if not root.exists():
            raise FileNotFoundError(
                f"Data directory '{root}' does not exist."
            )

        if not root.is_dir():
            raise NotADirectoryError(
                f"'{root}' is not a directory."
            )

        self.skipped.clear()

        documents: list[Document] = []

        for path in sorted(root.rglob("*")):

            if not path.is_file():
                continue

            relative_path = str(path.relative_to(root))

            if path.suffix.lower() not in self.SUPPORTED:
                self.skipped.append(relative_path)
                continue

            try:
                loaded_docs = self._load_file(path)

                for doc in loaded_docs:

                    if not doc.page_content.strip():
                        continue

                    doc.metadata.update({
                        "source": relative_path,
                        "file_name": path.name,
                        "file_type": path.suffix.lower(),
                    })

                    documents.append(doc)

                if not loaded_docs or not any(
                    doc.page_content.strip()
                    for doc in loaded_docs
                ):
                    self.skipped.append(relative_path)

            except Exception as exc:

                print(
                    f"  ! Could not read {relative_path}: {exc}"
                )

                self.skipped.append(relative_path)

        return documents

class Chunker:
    """Sentence-aware sliding-window chunker with character overlap."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                " ",
                "",
            ],
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: list[Document]) -> list[Document]:

        return self.splitter.split_documents(documents)

class IngestionPipeline:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.loader = DocumentLoader()
        self.chunker = Chunker(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

    # def run(self) -> IngestionReport:
    def run(self):
        start = time.time()
        docs = self.loader.load_dir(self.cfg.data_dir)
        if not docs:
            raise RuntimeError(f"No supported documents found in '{self.cfg.data_dir}'.")

        elapsed = time.time() - start
        
        print("\n" + "=" * 50)
        print(" INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f" Documents loaded : {len(docs)}")
        print(f" Execution time   : {elapsed:.2f} seconds")

        source_summary = defaultdict(lambda: {"count": 0, "is_pdf": False})

        for doc in docs:
            source = doc.metadata["source"]

            source_summary[source]["count"] += 1
            source_summary[source]["is_pdf"] = (
                doc.metadata.get("page") is not None
            )

        print("\nLoaded document sources:")
        print("-" * 60)

        for index, (source, info) in enumerate(source_summary.items(), start=1):
        
            if info["is_pdf"]:
                print(f"{index:02d}. {source} (Total Pages: {info['count']})")
            else:
                print(f"{index:02d}. {source}")

        print("=" * 60)

        split_docs = self.chunker.split(docs)

        chunks = []
        source_indices: dict[str, int] = {}

        for doc in split_docs:
            source = doc.metadata["source"]

            index = source_indices.get(source, 0)

            chunk = Document(
                id=f"{source}::{index}",
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "source": source,
                    "chunk_index": index,
                },
            )

            chunks.append(chunk)
            source_indices[source] = index + 1


        # Print chunk details
        print("\n" + "=" * 60)
        print(" CHUNKING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f" Total chunks generated : {len(chunks)}")

        # print("\nChunk Details:")
        # print("-" * 60)

        # for chunk in chunks:
        #     print(f"\nChunk ID: {chunk.id}")
        #     print(f"Source: {chunk.metadata['source']}")
        #     print(f"Chunk Index: {chunk.metadata['chunk_index']}")
        #     print(f"Text Length: {len(chunk.page_content)}")
        #     print(chunk.page_content)
        #     print("-" * 60)

        # print("=" * 60)

        texts = [chunk.page_content for chunk in chunks]

        vectors = self.cfg.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print("\n" + "=" * 60)
        print(" CHUNKING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f" Total embeddings generated : {len(vectors)}")
        print(f" Embedding shape            : {vectors.shape}")
        print(f" Embedding dimensions       : {vectors.shape[1]}")

        print("=" * 60)
        

def main() -> None:
    p = argparse.ArgumentParser(description="Build the RAG vector index.")
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--index-dir", default=Config.index_dir)
    p.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=Config.chunk_overlap)
    args = p.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    report = IngestionPipeline(cfg).run()

if __name__ == "__main__":
    main()
