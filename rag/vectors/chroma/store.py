from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class ChromaVectorStore:

    def __init__(
        self,
        persist_directory: str = "index/chroma",
        embedding_model: Embeddings | None = None,
        collection_name: str = "rag_documents",
    ):

        if embedding_model is None:
            raise ValueError("Embedding model is required.")

        self.persist_directory = Path(persist_directory)

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_model = embedding_model

        self.collection_name = collection_name

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.persist_directory),
            collection_metadata={
                "hnsw:space": "cosine"
            },
        )

    def add_document(
        self,
        document: Document
    ):
    
        if document.id is None:
            raise ValueError("Document ID is required.")

        self.db.add_documents(
            documents=[document],
            ids=[str(document.id)],
        )

    def add_documents(
        self,
        documents: list[Document],
    ):

        if not documents:
            return

        ids = [str(doc.id) for doc in documents]

        self.db.add_documents(
            documents=documents,
            ids=ids,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ):

        return self.db.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ):

        return self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict | None = None,
    ):

        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 5},
        )

    def delete(self, ids: list[str]):

        self.db.delete(ids=ids)

    def count(self) -> int:

        return self.db._collection.count()

    def get(self):

        return self.db

    def get_document(self, document_id: str) -> Document | None:
    
        result = self.db.get(
            ids=[document_id],
            include=["documents", "metadatas"],
        )
    
        if not result["ids"]:
            return None
    
        return Document(
            id=result["ids"][0],
            page_content=result["documents"][0],
            metadata=result["metadatas"][0] or {},
        )
    
    
    def get_all_documents(self) -> list[Document]:
    
        result = self.db.get(
            include=["documents", "metadatas"],
        )
    
        return [
            Document(
                id=result["ids"][i],
                page_content=result["documents"][i],
                metadata=result["metadatas"][i] or {},
            )
            for i in range(len(result["ids"]))
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:

        return self.db.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:

        kwargs = {
            "query": query,
            "k": k,
            "filter": filter,
        }

        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold

        return self.db.similarity_search_with_relevance_scores(
            **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:

        return self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
    ) -> list[Document]:

        return self.db.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
    ) -> list[Document]:

        return self.db.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:

        return self.db.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:

        return self.db.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def as_retriever(
        self,
        search_type: Literal[
            "similarity",
            "similarity_score_threshold",
            "mmr",
        ] = "similarity",
        search_kwargs: dict | None = None,
    ):

        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 5},
        )

    def search(
        self,
        query: str,
        search_type: Literal[
            "similarity",
            "relevance",
            "distance",
            "mmr",
        ] = "relevance",
        k: int = 5,
        filter: dict | None = None,
        score_threshold: float | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Any]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if search_type == "similarity":

            return self.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )

        elif search_type == "relevance":

            return self.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
                score_threshold=score_threshold,
            )

        elif search_type == "distance":

            return self.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )

        elif search_type == "mmr":

            return self.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )

        raise ValueError(
            f"Unsupported search type: {search_type}"
        )

    def search_many(
        self,
        queries: list[str],
        k: int = 5,
        search_type: str = "relevance",
    ) -> dict[str, list[Any]]:

        return {
            query: self.search(
                query=query,
                k=k,
                search_type=search_type,
            )
            for query in queries
        }