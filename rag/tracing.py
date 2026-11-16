import os
from decouple import config

def setup_tracing(project: str | None = None) -> bool:
    """
    python-decouple reads .env but does NOT put values into os.environ,
    while LangSmith only reads os.environ. So copy them across here.
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