from pathlib import Path
from dataclasses import dataclass
import argparse
import re
import time

from common import Config

@dataclass
class Document:
    source: str
    text: str

class DocumentLoader:
    """Reads .txt / .md / .pdf files from a directory (recursively)."""

    SUPPORTED = {".txt", ".md", ".pdf"}

    def __init__(self):
        self.skipped: list[str] = []

    @staticmethod
    def _read(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader  # optional dependency

            return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)

        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".md":  # strip heading markers, keep the heading text
            text = re.sub(r"^#{1,6}\s*(.+)$", r"\1.", text, flags=re.M)
        return text

    def load_dir(self, data_dir: str | Path) -> list[Document]:
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Data directory '{root}' does not exist.")
        docs = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            if path.suffix.lower() not in self.SUPPORTED:
                self.skipped.append(rel)
                continue
            try:
                text = self._read(path)
            except Exception as exc:  # keep ingesting even if one file is broken
                print(f"  ! could not read {rel}: {exc}")
                self.skipped.append(rel)
                continue
            if text.strip():
                docs.append(Document(source=rel, text=text))
            else:
                self.skipped.append(rel)
        return docs

class IngestionPipeline:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.loader = DocumentLoader()

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

        print("\nLoaded document sources:")
        print("-" * 60)

        for index, doc in enumerate(docs, start=1):
            print(f"{index:02d}. {doc.source}")

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
