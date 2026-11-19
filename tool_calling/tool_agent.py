from langchain_core.messages import HumanMessage, ToolMessage

from llm import LLM
from tools import (add_numbers,
            get_weather,
            get_user_info)

class ToolAgent:

    def __init__(self):

        # Get raw model
        self.llm = LLM().get_model()

        # Register tools
        self.tools = [
            add_numbers,
            get_weather,
            get_user_info
        ]

        self.tool_map = {
            tool.name: tool
            for tool in self.tools
        }

        # Bind tools
        self.llm_with_tools = self.llm.bind_tools(
            self.tools
        )

        self.max_iterations = 10

    def execute_tool(self, tool_call):

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"Executing: {tool_name}")
        print(f"Arguments: {tool_args}")

        selected_tool = self.tool_map.get(tool_name)

        if not selected_tool:
            return {"error": "Tool not found"}

        try:
            return selected_tool.invoke(tool_args)

        except Exception as e:
            return {"error": str(e)}

    def chat(self, user_input):

        messages = [
            HumanMessage(content=user_input)
        ]

        for _ in range(self.max_iterations):

            response = self.llm_with_tools.invoke(messages)
            print(f"Tool Bind LLM Response: {response.tool_calls}")

            messages.append(response)

            # No more tool calls
            if not response.tool_calls:
                return response.content

            # Execute all requested tools
            for tool_call in response.tool_calls:

                result = self.execute_tool(tool_call)

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

        return "Maximum tool execution iterations reached."


def main():

    agent = ToolAgent()

    response = agent.chat(
        """
        Calculate 10 + 20.
        Get the weather in Dhaka.
        """
    )
        # Get information about user 101.

    print("\nFinal Answer:")
    print(response)


if __name__ == "__main__":
    main()