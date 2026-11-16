import time
import argparse

from langchain_core.documents import Document

from vectors.faiss.store import FAISSVectorStore
from vectors.chroma.store import ChromaVectorStore
from common import Config, SentenceTransformerEmbeddings, LLM

from tracing import (setup_tracing)

setup_tracing()

class Retriever:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        k: int
    ) -> list[tuple[Document, float]]:

        return self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k
        )


class GenerationPipeline:

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()

        self.embedding_model = SentenceTransformerEmbeddings(
            self.cfg.embedding_model_name
        )

        self.vector_store = self._initialize_vector_store()

        self.retriever = Retriever(
            self.vector_store
        )

        self.llm = LLM()

    def _initialize_vector_store(self):

        if self.cfg.store_type == "chroma":

            return ChromaVectorStore(
                persist_directory=(
                    f"{self.cfg.index_dir}/chroma"
                ),
                embedding_model=self.embedding_model,
                collection_name=self.cfg.collection_name,
            )

        elif self.cfg.store_type == "faiss":

            return FAISSVectorStore(
                persist_directory=(
                    f"{self.cfg.index_dir}/faiss"
                ),
                embedding_model=self.embedding_model,
                index_name=self.cfg.collection_name
            )

        raise ValueError(
            "store_type must be either 'chroma' or 'faiss'"
        )

    def _build_context(
        self,
        documents: list[tuple[Document, float]]
    ) -> str:

        context_parts = []

        for i, (doc, score) in enumerate(documents, start=1):

            source = doc.metadata.get("source", "unknown")

            context_parts.append(
                f"""
                Document {i}
                Source: {source}
                Content:
                {doc.page_content}
                """
            )

        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        k: int | None = None,
    ) -> str:

        start = time.time()

        # 1. Retrieve relevant documents
        contexts = self.retriever.retrieve(
            query=query,
            k=k or self.cfg.top_k
        )

        retrieval_time = time.time() - start

        if not contexts:
            return "No relevant documents found."

        # 2. Build context
        context = self._build_context(contexts)


        print("\n" + "=" * 80)
        print("RETRIEVED CONTEXT DETAILS")
        print("=" * 80)

        print(f"Context loaded: {len(contexts)}")
        print(f"Execution time: {retrieval_time:.2f} seconds")

        print("-" * 80)

        for i, (doc, score) in enumerate(contexts, start=1):

            print(f"\nContext #{i}")
            print(f"ID: {doc.id or doc.metadata.get('id', 'N/A')}")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Score: {score:.4f}")
            print(f"Content length: {len(doc.page_content)}")
            print(f"Content: {doc.page_content}")
        
        # 3. Generate answer
        generation_start = time.time()

        answer = self.llm.generate(
            question=query,
            context=context
        )

        generation_time = time.time() - generation_start

        print("\n" + "=" * 80)
        print("GENERATED ANSWER")
        print("=" * 80)

        print(answer)

        print("\n" + "-" * 80)
        print(f"Retrieval time: {retrieval_time:.2f}s")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Total time: {time.time() - start:.2f}s")

        return answer


def main() -> None:

    p = argparse.ArgumentParser(
        description="Ask questions against the RAG index."
    )

    p.add_argument(
        "--query",
        help="Single question; omit for interactive mode"
    )

    p.add_argument(
        "--index-dir",
        default=Config().index_dir
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=Config().top_k
    )

    args = p.parse_args()

    cfg = Config(
        index_dir=args.index_dir,
        top_k=args.top_k
    )

    pipeline = GenerationPipeline(cfg)
    # pipeline.generate("give me course list with course code and credit score of year 1 term 2")
    if args.query:

        pipeline.generate(args.query)

    else:

        while True:

            query = input("\nAsk a question (or 'exit'): ").strip()

            if query.lower() in {"exit", "quit"}:
                break

            if not query:
                continue

            pipeline.generate(query)


if __name__ == "__main__":
    main()
