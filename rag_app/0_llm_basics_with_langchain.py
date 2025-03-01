from configuration import model, model_provider, user
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from error_handler import handle_exception


def main() -> None:
    llm = init_chat_model(model=model, model_provider=model_provider)

    messages = [
        ("human", 
        """You are a helpful and insightfull butler name Alfred working for {user}. Keep your answer short.
        The only thing you know about our current state is the conversation history, do no assume anything.
        If conversion history is empty, we are starting a new conversation and you should remember what we talk about.
        Do not invent anything that is not inside our conversation or you knowledge of the world.
        - conversion history messages (your answer have the word 'model' before them): {history} 
        - last message: {message}
        """)
    ]

    prompt = ChatPromptTemplate(messages)

    # response = llm.invoke(prompt.invoke({"user": "enrique"}))

    # print(f"model answer: {response.content}")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting interactive chat.")
            break
        formated_message = prompt.invoke({"user": user, "history": "\n".join(history), "message": user_input}) 
        # response = llm.invoke(formated_message)
        print("model answer: ", end="")
        # streaming the answer
        response = ""
        for chunk in llm.stream(formated_message):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        print()
        history.append(f"user: {user_input}")
        history.append(f"model: {response}")

if __name__ == "__main__":
    # Run the pipeline test when this script is executed.
    try:
        main()
    except Exception as e:
        handle_exception(e, model)