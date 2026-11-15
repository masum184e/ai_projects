import faiss
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

class FAISSVectorStore:

    def __init__(
        self,
        persist_directory: str = "index/faiss",
        embedding_model: Embeddings | None = None,
        index_name: str = "index"
    ):

        if embedding_model is None:
            raise ValueError("Embedding model is required.")

        self.persist_directory = Path(persist_directory)

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_model = embedding_model

        self.index_name = index_name

        self.index_path = (
            self.persist_directory / f"{self.index_name}.faiss"
        )

        if self.index_path.exists():

            self.db = FAISS.load_local(
                folder_path=str(self.persist_directory),
                embeddings=self.embedding_model,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )

        else:

            dimension = len(
                self.embedding_model.embed_query("initialization")
            )

            index = faiss.IndexFlatIP(dimension)

            self.db = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )

    def add_document(
        self,
        document: Document,
    ):
    
        if document.id is None:
            raise ValueError("Document ID is required.")

        self.db.add_documents(
            documents=[document],
            ids=[str(document.id)],
        )

        self.save()

    def add_documents(
        self,
        documents: list[Document],
    ):

        if not documents:
            return

        if any(doc.id is None for doc in documents):
            raise ValueError("All documents must have IDs.")

        ids = [str(doc.id) for doc in documents]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate document IDs found.")

        self.db.add_documents(
            documents=documents,
            ids=ids,
        )

        self.save()

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:

        return self.db.similarity_search(
            query=query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:

        return self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
        )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
        score_threshold: float | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:

        results = self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
        )

        # IndexFlatIP with normalized vectors returns
        # cosine similarity in the range [-1, 1].
        scored_results = [
            (doc, (float(score) + 1.0) / 2.0)
            for doc, score in results
        ]

        if score_threshold is not None:
            scored_results = [
                (doc, score)
                for doc, score in scored_results
                if score >= score_threshold
            ]

        return scored_results

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

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:

        return self.db.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:

        return self.db.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
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

        if fetch_k < k:
            raise ValueError("fetch_k must be >= k.")

        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1.")

        if search_type == "similarity":

            return self.similarity_search(
                query=query,
                k=k,
                filter=filter,
                fetch_k=fetch_k,
            )

        elif search_type == "relevance":

            return self.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
                score_threshold=score_threshold,
                fetch_k=fetch_k,
            )

        elif search_type == "distance":

            # For IndexFlatIP, these are raw inner-product
            # similarity scores, not distances.
            return self.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
                fetch_k=fetch_k,
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

    def delete(self, ids: list[str]):

        self.db.delete(ids=ids)

        self.save()

    def save(self):

        self.db.save_local(
            folder_path=str(self.persist_directory),
            index_name=self.index_name,
        )

    def count(self) -> int:

        return self.db.index.ntotal

    def get(self):

        return self.db

    def get_document(
        self,
        document_id: str,
    ) -> Document | None:
    
        docstore = self.db.docstore

        document = docstore.search(str(document_id))

        if isinstance(document, Document):
            return document

        return None


    def get_all_documents(self) -> list[Document]:

        documents = []

        for document_id in self.db.index_to_docstore_id.values():

            document = self.db.docstore.search(document_id)

            if isinstance(document, Document):
                documents.append(document)

        return documents