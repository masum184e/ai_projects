import time
import argparse

from langchain_core.documents import Document

from vectors.faiss.store import FAISSVectorStore
from vectors.chroma.store import ChromaVectorStore

from common import (Config, SentenceTransformerEmbeddings, LLM, Reranker)
from tracing import (setup_tracing, traceable, drop_self, docs_to_trace)

setup_tracing()

class Retriever:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    @traceable(
        run_type="retriever", name="vector_search",
        process_inputs=drop_self, process_outputs=docs_to_trace,
    )
    def retrieve(
        self,
        query: str,
        retriever_k: int
    ) -> list[tuple[Document, float]]:

        return self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=retriever_k,
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

        self.reranker = Reranker(self.cfg.reranker_model_name)
        
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


    def _retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:

        final_top_k = top_k if top_k is not None else self.cfg.top_k

        retriever_k = max(self.cfg.retriever_k, final_top_k)

        candidates = self.retriever.retrieve(
            query=query,
            retriever_k=retriever_k,
        )

        return self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=final_top_k,
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

    @traceable(
        run_type="chain", name="rag_query",
        process_inputs=drop_self, tags=["rag"],
    )
    def generate(
        self,
        query: str,
        top_k: int | None = None,
    ) -> str:

        start = time.time()

        contexts = self._retrieve(
            query=query,
            top_k=top_k,
        )

        retrieval_time = time.time() - start

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
            print(f"Rerank score: {score:.4f}")
            print(f"Vector score: {doc.metadata.get('retrieval_score', float('nan')):.4f}")
            print(f"Content length: {len(doc.page_content)}")
            print(f"Content: {doc.page_content}")

        # 3. Generate answer
        generation_start = time.time()

        answer = self.llm.generate(
            question=query,
            context=contexts
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

    p.add_argument(
        "--retriever-k",
        type=int,
        default=Config().retriever_k,
        help="Candidates fetched from the vector store before reranking"
    )

    args = p.parse_args()

    cfg = Config(
        index_dir=args.index_dir,
        top_k=args.top_k,
        retriever_k=args.retriever_k,
    )

    pipeline = GenerationPipeline(cfg)

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