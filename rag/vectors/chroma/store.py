from pathlib import Path

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