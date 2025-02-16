from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from final_step_vector_db import create_vector_db

model = "llama3.1:8b"
embedding_model = "llama3.1:8b"
model_provider = "ollama"
user = "Antonio Badilla-Olivas"


llm = init_chat_model(model, model_provider=model_provider)
db = create_vector_db()


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a queary"""
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.01},
    )
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState) -> dict:
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        f"""
        You are {user}'s personal assistant.
        You are in charge of question-answering tasks related to their experience with Software Engineering and Computer Science.
        Use the following pieces of retrieved context to answer the question.
        If asked, you can answer questions about their life, interests, and hobbies if sources allow.
        The context is taken from {user}'s CV, LinkedIn, and other relevant documents.
        If you don't know the answer, just say that you don't know. Your statements must be supported by the sources.
        Use no more than 5 sentences and be friendly.
        """
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_rag_graph() -> tuple:
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(ToolNode([retrieve]))
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}
    return graph, config
