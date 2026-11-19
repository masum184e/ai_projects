import numpy as np
from decouple import config
from langchain_groq import ChatGroq
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tracing import traceable, embed_inputs, embed_outputs, drop_self, docs_to_trace

class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @traceable(
        run_type="embedding", name="embed_documents",
        process_inputs=embed_inputs, 
        process_outputs=embed_outputs
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vectors.astype(np.float32).tolist()

    @traceable(
        run_type="embedding", name="embed_query",
        process_inputs=embed_inputs, 
        process_outputs=embed_outputs
    )
    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector.astype(np.float32).tolist()

    def get_model(self):
        return self.model    

@dataclass
class Config:
    data_dir: str = "docs"

    chunk_size: int = 1200
    chunk_overlap: int = 120

    embedding_model_name: str = "all-MiniLM-L6-v2"

    store_type: str = "faiss"
    index_dir: str = "index"
    collection_name: str = "rag_documents"

    retriever_k: int = 5
    top_k: int = 2
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class LLM:

    def __init__(
        self,
        model_name: str = "qwen/qwen3.8-27b",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):

        self.model_name = model_name
        
        print(f"Using Groq model: {model_name}")
        
        self.model = ChatGroq(
            model=model_name,
            api_key=config("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.output_parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a helpful AI assistant specialized in
                answering questions using retrieved context.

                Follow these rules:
                1. Answer using only the provided context.
                2. Do not make up information.
                3. If the answer is not available in the context,
                   clearly say that you don't know.
                4. Keep answers clear, accurate, and relevant.
                5. Do not unnecessarily repeat information.

                Context:
                {context}
                """
            ),
            (
                "human",
                "{question}"
            )
        ])

        self.chain = self.prompt | self.model | self.output_parser

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        response = self.chain.invoke(
            {"question": question, "context": context},
            config={
                "run_name": "groq_generate",
                "tags": ["llm", "groq"],
                "metadata": {"model": self.model_name, "temperature": self.model.temperature},
            },
        )

        return response

    def stream(
        self,
        question: str,
        context: str,
    ):

        for chunk in self.chain.stream({
            "question": question,
            "context": context,
        }):
            yield chunk

    def get_model(self):
        """Return the underlying LangChain chat model."""
        return self.model

class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ):
        print(f"Using reranker model: {model_name}")

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        self.batch_size = batch_size

    @traceable(
        run_type="retriever", name="rerank",
        process_inputs=drop_self, process_outputs=docs_to_trace,
    )
    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc, _ in documents]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results = []

        for (doc, retrieval_score), rerank_score in ranked:
            reranked_doc = Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "retrieval_score": float(retrieval_score),
                    "rerank_score": float(rerank_score),
                },
            )
            results.append((reranked_doc, float(rerank_score)))

        return results