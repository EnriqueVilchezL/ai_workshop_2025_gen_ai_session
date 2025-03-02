"""
Workshop Guide: Building a Retrieval-Augmented Generation (RAG) Pipeline – Part 3

This file demonstrates an end-to-end RAG system:
1. Loading and processing documents (PDFs and text files) from the 'my_sources' folder.
2. Splitting documents into smaller chunks and annotating them.
3. Creating (or loading) a vector database (via Chroma) for storing document embeddings.
4. Initializing a Language Model (LLM) to use in the pipeline.
5. Defining a retrieval tool that fetches context based on a query.
6. Building a state graph that sequences retrieval and answer generation.
7. Running an interactive command-line chat to demonstrate the complete pipeline.
"""

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

from configuration import (
    embedding_model,
    model,
    model_provider,
    user,
    db_dir,
    sources_dir,
    db_search_kwargs,
    db_search_type,
)
from error_handler import handle_exception


def load_documents_from_my_sources(sources_path: Path) -> list:
    """
    Loads and processes documents from the specified sources directory.

    Steps:
    1. Iterates over files with .pdf or .txt extensions.
    2. Uses the proper loader (TextLoader for .txt, PyPDFLoader for .pdf) to load the document.
    3. Adds metadata (source file name) to each document.
    4. Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
    5. Divides the chunks into sections (beginning, middle, end) for better context annotation.

    Args:
        sources_path (Path): Path to the directory containing source documents.

    Returns:
        list: A list of processed document chunks.
    """
    # Identify candidate source files.
    source_documents = [
        {"file": f, "extension": f.suffix}
        for f in sources_path.iterdir()
        if f.suffix in {".pdf", ".txt"}
    ]

    docs_parsed = []
    # Process each file.
    for source in source_documents:
        file_path = source["file"]
        # Load the document using the appropriate loader.
        match source["extension"]:
            case ".txt":
                loader = TextLoader(file_path)
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case _:
                continue  # Skip unsupported file extensions.
        documents = loader.load()
        # Add metadata indicating the source of the document.
        for doc in documents:
            doc.metadata = {"source": source["file"].name}
        docs_parsed.extend(documents)

    # Split documents into chunks for better handling by the LLM.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum number of characters in each chunk.
        chunk_overlap=100,  # Overlap between chunks for continuity.
        add_start_index=True,  # Try to preserve the original order.
    )
    all_splits = text_splitter.split_documents(docs_parsed)
    total_documents = len(all_splits)
    third = total_documents // 3

    # Annotate each document chunk with its relative section in the original document.
    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"
    return all_splits


def create_vector_db() -> Chroma:
    """
    Creates or loads a persistable vector database using Chroma.

    Steps:
    1. Determines the current directory and the location of source files and persistent storage.
    2. Generates embeddings for documents using OllamaEmbeddings.
    3. If a persistent database exists, it loads the existing one.
    4. Otherwise, it loads documents from the sources, processes them, and creates a new vector database.

    Returns:
        Chroma: A vector database instance ready for retrieval-based queries.
    """
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / sources_dir
    persistent_directory = current_dir / db_dir

    embeddings = OllamaEmbeddings(model=embedding_model)
    if persistent_directory.exists():
        print("Using existing db")
        return Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print("Persistent directory does not exist. Initializing vector store...")
    # Ensure that the source directory exists.
    if not sources_path.exists():
        raise FileNotFoundError(
            f"The directory {sources_path} does not exist. Please check the path."
        )
    docs = load_documents_from_my_sources(sources_path)
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(persistent_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


# -------------------------------
# Global Instances: Initialize the LLM and vector database.
llm = init_chat_model(model, model_provider=model_provider)
db = create_vector_db()


# -------------------------------
# TOOL: retrieve
# This function is decorated as a tool that can be dynamically bound and called in the pipeline.
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve information related to a query.

    This function uses the vector database retriever to find relevant document chunks that match the query.
    It returns a serialized string containing source metadata and document content, along with the raw documents.

    Args:
        query (str): The user's query.

    Returns:
        tuple: A tuple containing a serialized string and the list of retrieved documents.
    """
    retriever = db.as_retriever(
        search_type=db_search_type,
        search_kwargs=db_search_kwargs,
    )
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState) -> dict:
    """
    Generates a tool call for either retrieval or direct response if no retrieval is needed.

    It binds the language model with the retrieval tool and invokes the LLM with the conversation state.

    Args:
        state (MessagesState): The current state containing the conversation messages.

    Returns:
        dict: A dictionary with a list of messages as the response.
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: MessagesState):
    """
    Generates an answer based on the retrieved context from prior tool invocations.

    Steps:
    1. Collects recent tool messages from the conversation state.
    2. Formats the collected document contexts into a single prompt.
    3. Creates a system message incorporating the user instructions and document context.
    4. Appends non-tool messages from the conversation history.
    5. Invokes the LLM using the formatted prompt to generate an answer.

    Args:
        state (MessagesState): The state payload containing conversation messages.

    Returns:
        dict: A dictionary containing a list of messages with the final answer.
    """
    # Collect tool messages in reverse order and then reverse them back.
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Create a unified string with the content from each tool message.
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        f"""
            You are {user}'s personal assistant.
            You are in charge of question-answering tasks related to their experience with Software engineering and Computer Science.
            Use the following pieces of retrieved context to answer the question. 
            The context is taken from {user}'s CV, Linkedin and other relevant documents.
            If you don't know the answer, just say that you don't know. Sources must sustain your words. Use 4 sentences maximum and keep the answer concise.
            """
        "\n\n"
        f"{docs_content}"
    )
    # Filter conversation messages to include only human, system or non-tool AI messages.
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in {"human", "system"}
        or (message.type == "ai" and not message.tool_calls)
    ]
    # Build the final prompt by prepending the system message.
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Call the LLM to generate the final output.
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_rag_graph() -> tuple:
    """
    Constructs the RAG pipeline graph which sequences retrieval and answer generation.

    Steps:
    1. Adds nodes for query_or_respond, tool-based retrieval, and answer generation.
    2. Sets the entry point for the pipeline.
    3. Adds conditional edges to redirect from query processing to tool retrieval.
    4. Connects the nodes to ensure sequential processing.
    5. Attaches a memory saver for checkpointing the graph state.

    Returns:
        tuple: The compiled graph and a configuration dictionary.
    """
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


def interactive_chat() -> None:
    """
    Starts an interactive chat session on the terminal.

    The session:
    1. Builds the RAG graph.
    2. Enters a loop to prompt the user for input.
    3. Streams the LLM response through the pipeline.
    4. Utilizes pretty-printing to display each output step.
    5. Exits when the user types 'exit' or 'quit'.
    """
    graph, config = build_rag_graph()
    print("Interactive conversation started. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting interactive chat.")
            break

        # Stream the answer for the given user query.
        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            # Pretty-print the last message of the current step.
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    try:
        interactive_chat()
    except Exception as e:
        handle_exception(e, model)
