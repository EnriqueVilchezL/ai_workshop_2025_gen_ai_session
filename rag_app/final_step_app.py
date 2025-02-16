import gradio as gr
from final_step_graph import build_rag_graph, user

# Build the graph and get the configuration
graph, config = build_rag_graph()

def gradio_chat(user_message, history):
    input_payload = {"messages": [{"role": "user", "content": user_message}]}
    
    history = history or []
    history.append((user_message, ""))
    
    # Accumulate the LLM response by processing the streamed pipeline steps.
    current_ai_message = ""
    for step in graph.stream(input_payload, stream_mode="values", config=config):
        # Print the current pipeline state to the terminal for debugging
        print("Pipeline step:", step)
        # Get the latest message from the current stream step
        message = step["messages"][-1]
        message.pretty_print()
        if getattr(message, "type", None) in ("human", "ai", "system"):
            current_ai_message = message.content

    # Update history with the final/current LLM message only
    history[-1] = (user_message, current_ai_message)
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown(f"## {user}'s Personal Assistant")
    gr.Markdown(f"Hi this is {user}'s personal assistant, I can offer answers on their skills, education and interests. :D")
    chatbot = gr.Chatbot()
    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here and press Enter"
        )
    user_input.submit(gradio_chat, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

demo.launch(share=True)