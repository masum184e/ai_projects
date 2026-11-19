from decouple import config
from langchain_groq import ChatGroq

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

    def get_model(self):
        """Return the underlying LangChain chat model."""
        return self.model