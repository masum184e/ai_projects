import numpy as np
from decouple import config
from langchain_groq import ChatGroq
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tracing import traceable, embed_inputs, embed_outputs

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

        return np.array(vectors, dtype=np.float32)

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

        return np.array(vector, dtype=np.float32)
        
@dataclass
class Config:
    data_dir: str = "docs"

    chunk_size: int = 1200
    chunk_overlap: int = 120

    embedding_model_name: str = "all-MiniLM-L6-v2"

    store_type: str = "faiss"
    index_dir: str = "index"
    collection_name: str = "rag_documents"

    top_k: int = 4

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