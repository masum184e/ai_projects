from langchain.agents import create_agent

from llm import LLM
from tools import (add_numbers,
            get_weather,
            get_user_info)

class CreateAgent:

    def __init__(self):

        # Get raw model
        self.model = LLM().get_model()

        # Register tools
        self.tools = [
            add_numbers,
            get_weather,
            get_user_info
        ]

        # Create agent
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=(
                "You are a helpful assistant. "
                "Use available tools when necessary. "
                "Combine tool results into a clear answer."
            )
        )

    def chat(self, user_input: str) -> str:

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config={
                "recursion_limit": 20
            }
        )

        return result["messages"][-1].content


def main():

    agent = CreateAgent()

    response = agent.chat(
        """
        Calculate 10 + 20.
        Get the weather in Dhaka.
        Get information about user 101.
        """
    )

    print("\nFinal Answer:")
    print(response)


if __name__ == "__main__":
    main()