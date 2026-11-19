from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # Mock weather API
    return f"Weather in {city}: Sunny, 30°C"


@tool
def get_user_info(user_id: str) -> dict:
    """Get user information using user ID."""
    users = {
        "101": {"name": "John", "age": 25},
        "102": {"name": "Alice", "age": 30}
    }

    return users.get(user_id, {"error": "User not found"})
