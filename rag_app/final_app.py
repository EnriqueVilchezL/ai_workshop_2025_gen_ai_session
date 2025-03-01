"""
Workshop Guide: Building a Gradio-based RAG Application – Final Step

This file demonstrates how to build an interactive Gradio application that:
1. Integrates with a pre-built RAG pipeline defined in rag_final_step_graph.py.
2. Streams responses generated by the language model.
3. Displays the final answer in a conversational UI.
4. Prints debugging output from each pipeline step to the terminal.
"""

import gradio as gr
from final_graph import build_rag_graph, user  # Import the graph builder and user name
from error_handler import handle_exception
from configuration import model

# Build the RAG pipeline graph and retrieve configuration settings.
graph, config = build_rag_graph()


def gradio_chat(user_message, history):
    """
    Process a user message and update the chat history.

    This function:
      1. Prepares an input payload for the RAG pipeline.
      2. Appends a new conversation turn with an empty AI response.
      3. Streams the pipeline's processing steps, printing each step for debugging.
      4. From the streamed steps, accumulates the final LLM response.
      5. Updates the chat history with the final answer.

    Args:
        user_message (str): The message input by the user.
        history (list): List of previous chat messages.

    Returns:
        tuple: Updated chat history and an empty string (clearing the input box).
    """
    # Structure the input for the pipeline using the user's message.
    input_payload = {"messages": [{"role": "user", "content": user_message}]}

    # Ensure the chat history is initialized and add the new message turn.
    history = history or []
    history.append((user_message, ""))

    # Initialize an empty string to accumulate the final LLM response.
    current_ai_message = ""
    # Process the pipeline stream step by step.
    for step in graph.stream(input_payload, stream_mode="values", config=config):
        # For debugging: print the complete pipeline state to the terminal.
        print("Pipeline step:", step)
        # Retrieve the latest message from the current step.
        message = step["messages"][-1]
        # Pretty-print the message details for clarity.
        message.pretty_print()
        # Only consider messages of type 'human', 'ai', or 'system' for the final output.
        if getattr(message, "type", None) in ("human", "ai", "system"):
            current_ai_message = message.content

    # Update the last entry in the history with the final LLM response.
    history[-1] = (user_message, current_ai_message)
    return history, ""


def main() -> None:
    # -------------------------------
    # Build the Gradio user interface.
    with gr.Blocks() as demo:
        # Display a markdown header with emojis for a friendlier look.
        gr.Markdown(f"## {user}'s Personal Assistant 🤖✨")
        # Provide a brief description of the assistant's capabilities with emojis.
        gr.Markdown(
            f"Hi there! 👋 This is {user}'s personal assistant. I can offer answers about skills, education, interests, and more! 😎"
        )
        # Create the chatbot component for displaying the conversation in messages format.
        chatbot = gr.Chatbot()
        # Create a horizontal row for the text input.
        with gr.Row():
            user_input = gr.Textbox(
                show_label=False,
                placeholder="Type your message here and press Enter 💬",
            )
        # Bind the textbox submission to the gradio_chat function,
        # updating the chatbot and clearing the input.
        user_input.submit(
            gradio_chat, inputs=[user_input, chatbot], outputs=[chatbot, user_input]
        )

    # Launch the Gradio app with sharing enabled.
    demo.launch(share=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e, model)
