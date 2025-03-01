"""
Workshop Example: Building an Interactive Chat with LangChain Agents and Tools

This file demonstrates how to:
1. Define custom tools using the @tool decorator
2. Create an agent that can use these tools
3. Set up an interactive chat interface with the agent
4. Stream responses for a better user experience
"""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
import random
from datetime import datetime
from configuration import model, model_provider, user
from error_handler import handle_exception


# Define some dummy tools for our agent
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression (e.g., '2 + 2', '3 * 4', '10 / 2').
        
    Returns:
        The result of evaluating the expression.
    """
    try:
        # Using eval is generally unsafe, but acceptable for this demo with known input
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a location (dummy data for demo purposes).
    
    Args:
        location: The city or location to get weather for.
        
    Returns:
        A string describing the current weather.
    """
    # Generate dummy weather data
    conditions = ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"]
    temperatures = range(0, 35)
    
    condition = random.choice(conditions)
    temperature = random.choice(temperatures)
    
    return f"Weather in {location}: {condition}, {temperature}°C"


@tool
def tell_joke() -> str:
    """
    Tells a random joke.
    
    Returns:
        A random joke.
    """
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a fish with no eyes? Fsh!",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus!",
        "How do you organize a space party? You planet!"
    ]
    return random.choice(jokes)


@tool
def get_current_time() -> str:
    """
    Get the current time.
    
    Returns:
        The current time in ISO format.
    """
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def main() -> None:
    """
    Run the interactive agent chat application.
    """
    # Initialize the language model
    llm = init_chat_model(model=model, model_provider=model_provider)
    
    # List all tools that will be available to the agent
    tools = [calculator, get_weather, tell_joke, get_current_time]
    
    # Create the prompt that defines the agent's behavior
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Alfred, a helpful assistant for {user}.
        You have access to the following tools: {tool_names}
        
        {tools}
        
        To use a tool, use the following format:
        ```
        Action: tool_name
        Action Input: the input to the tool
        ```
        
        The observation will be the result of the tool.
        
        After you use a tool or if you don't need to use a tool, respond directly to the user.
        Always prioritize being helpful and concise.
        """),
        ("human", "{input}"),
        ("human", "{agent_scratchpad}")
    ])
    
    # Create the agent
    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )
    
    # Create the agent executor which will run the agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Start the interactive chat session
    print("🤖 Alfred at your service! Ask me anything or type 'exit' to quit.")
    history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Alfred: Goodbye! Have a wonderful day.")
            break

        # Include conversation history for context
        full_input = user_input
        if history:
            full_input = "Previous conversation:\n" + "\n".join(history) + "\n\nCurrent question: " + user_input
        
        print("\nAlfred: ", end="")
        
        # Execute the agent and stream the response
        response = ""
        # Note: AgentExecutor doesn't support native streaming, but we can print steps
        result = agent_executor.invoke({"input": full_input, "user": user})
        
        # Print the final answer
        print(result["output"])
        response = result["output"]
        
        # Store conversation in history
        history.append(f"User: {user_input}")
        history.append(f"Alfred: {response}")
        
        # Limit history length to prevent context overflow
        if len(history) > 10:
            history = history[-10:]


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e, model)