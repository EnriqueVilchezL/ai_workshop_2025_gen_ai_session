from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

def load_documents_from_my_sources(sources_path: Path) -> list:
    # Load documents
    source_documents = [
        {"file": f, "extension": f.suffix}
        for f in sources_path.iterdir()
        if f.suffix in {".pdf", ".txt"}
    ]

    docs_parsed = []
    for source in source_documents:
        file_path = sources_path / source["file"]
        match source["extension"]:
            case ".txt":
                loader = TextLoader(file_path)
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case _:
                continue
        documents = loader.load()
        for doc in documents:
            doc.metadata = {"source": source["file"].name}
        docs_parsed.extend(documents)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs_parsed)
    total_documents = len(all_splits)
    third = total_documents // 3

    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"
    return all_splits


def create_vector_db() -> Chroma:
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / "my_sources"
    persistent_directory = current_dir / "db" / "antonio_db"

    embeddings = OllamaEmbeddings(model=embedding_model)
    if persistent_directory.exists():
        print("Using existing db")
        return Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print("Persistent directory does not exist. Initializing vector store...")
    # Ensure the sources directory exists
    if not sources_path.exists():
        raise FileNotFoundError(
            f"The directory {sources_path} does not exist. Please check the path."
        )
    docs = load_documents_from_my_sources(sources_path)
    # Display information about the split documents
    # print("\n--- Document Chunks Information ---")
    # print(f"Number of document chunks: {len(docs)}")
    # print(f"Sample chunk:\n{docs[0].page_content}\n")
    #
    # print("\n--- Creating vector store ---")
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(persistent_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


model = "llama3.1:8b"
embedding_model = "llama3.1:8b"
model_provider = "ollama"


llm = init_chat_model(model, model_provider=model_provider)
db = create_vector_db()
user = "Antonio Badilla-Olivas"

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


tools = ToolNode([retrieve])


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
            You are an {user}'s personal assistant.
            You are in charge of question-answering tasks related to their experience with Software engineering and Computer Science.
            Use the following pieces of retrieved context to answer the question. 
            The context is taken from {user}'s CV, Linkedin and other relevant documents.
            If you don't know the answer, just say that you don't know. Sources must sustain your words. Use 4 sentences maximum and keep the answer concise.
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
    graph_builder.add_node(tools)
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

def interactive_chat() -> None:
    graph, config = build_rag_graph()
    print("Interactive conversation started. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting interactive chat.")
            break

        # stream the answer for the given user question
        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print() 

if __name__ == "__main__":
    interactive_chat()
