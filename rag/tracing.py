import os
from decouple import config
from langsmith import traceable

def setup_tracing(project: str | None = None) -> bool:
    """
    python-decouple reads .env but does NOT put values into os.environ,
    while LangSmith only reads os.environ. So copy them across here.
    LangSmith trace only LLM call without any configuration
    """
    
    enabled = str(config("LANGSMITH_TRACING", default="true")).lower() == "true"
    api_key = config("LANGSMITH_API_KEY", default="")
    
    if not enabled or not api_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        print("LangSmith tracing: DISABLED")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project or config(
        "LANGSMITH_PROJECT", default="rag-project"
    )
    endpoint = config("LANGSMITH_ENDPOINT", default="")
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    print(f"LangSmith tracing: ENABLED (project={os.environ['LANGSMITH_PROJECT']})")
    return True

def embed_inputs(inputs: dict) -> dict:
    if "texts" in inputs:
        return {"num_texts": len(inputs["texts"])}
    return {"text": inputs.get("text")}

def embed_outputs(vectors) -> dict:
    shape = getattr(vectors, "shape", None)
    return {"shape": list(shape) if shape is not None else len(vectors)}

def drop_self(inputs: dict) -> dict:
    """Remove `self` so objects (models, DB handles) aren't serialized."""
    return {k: v for k, v in inputs.items() if k != "self"}

def docs_to_trace(pairs) -> dict:
    """
    Format retriever output the way LangSmith's retriever view expects:
    {"documents": [{"page_content", "type": "Document", "metadata"}]}
    """
    documents = []
    for doc, score in pairs or []:
        documents.append({
            "page_content": doc.page_content,
            "type": "Document",
            "metadata": {
                **doc.metadata,
                "id": doc.id,
                "relevance_score": round(float(score), 4),
            },
        })
    return {"documents": documents}