import faiss
from pathlib import Path

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
                normalize_L2=False,
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

        ids = [str(doc.id) for doc in documents]

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