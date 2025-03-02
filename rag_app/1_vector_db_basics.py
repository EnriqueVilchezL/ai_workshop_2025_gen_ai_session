"""
Workshop Guide: Building a Retrieval-Augmented Generation (RAG) Pipeline

This file demonstrates how to:
1. Load and preprocess documents (PDFs or TXT files).
2. Split documents into manageable chunks.
3. Create a vector database (using Chroma) to store document embeddings.
4. Build a prompt template for an LLM-based Q&A system.
5. Construct a simple pipeline (graph) that retrieves context and generates an answer.
6. Test the pipeline by invoking a sample query.
"""

from pathlib import Path

# Import text splitter for breaking documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import Chroma to manage vector databases
from langchain_chroma import Chroma

# Import loaders to load PDF and text files
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Import embeddings model interface for generating embeddings
from langchain_ollama import OllamaEmbeddings

# Import initializer for the chat model (LLM)
from langchain.chat_models import init_chat_model

# Import types for annotations and type checking
from typing_extensions import List, TypedDict

# Import the Document class to represent text documents with metadata
from langchain_core.documents import Document

# Import prompt template for chat based LLMs
from langchain_core.prompts import ChatPromptTemplate

# Import graph utilities to build a sequence of steps in the pipeline
from langgraph.graph import START, StateGraph
from configuration import (
    embedding_model,
    model,
    model_provider,
    user,
    db_dir,
    sources_dir,
    db_search_type,
    db_search_kwargs,
)
from error_handler import handle_exception


def load_documents_from_my_sources(sources_path: Path) -> list:
    """
    Loads and parses documents from the specified sources directory.

    1. Iterates over files in 'sources_path' that have .pdf or .txt extensions.
    2. Uses appropriate loaders (TextLoader or PyPDFLoader) based on the file extension.
    3. Attaches source metadata to each document.
    4. Splits each document into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        sources_path (Path): The path to the directory with source documents.

    Returns:
        list: A list of document chunks.
    """
    # List all files with extensions .pdf or .txt
    source_documents = [
        {"file": f, "extension": f.suffix}
        for f in sources_path.iterdir()
        if f.suffix in {".pdf", ".txt"}
    ]

    docs_parsed = []
    # Loop through each candidate file, load and parse its contents.
    for source in source_documents:
        file_path = source["file"]
        # Choose the loader based on the file extension
        match source["extension"]:
            case ".txt":
                loader = TextLoader(file_path)
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case _:
                continue
        documents = loader.load()
        # Add metadata indicating the source file name
        for doc in documents:
            doc.metadata = {"source": source["file"].name}
        docs_parsed.extend(documents)

    # Use a text splitter to break documents into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    return text_splitter.split_documents(docs_parsed)


def create_vector_db() -> Chroma:
    """
    Creates or loads a vector database using Chroma.

    This function:
    1. Determines the directories for sources and persistent storage.
    2. Instantiates an embeddings generator using OllamaEmbeddings.
    3. Checks if a persistent vector database already exists:
       - If yes, it loads the existing database.
       - If not, it loads and splits the source documents and creates a new vector database.
    4. Returns an instance of the vector database (Chroma).

    Returns:
        Chroma: The vector database instance.
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
    if not sources_path.exists():
        raise FileNotFoundError(
            f"The directory {sources_path} does not exist. Please check the path."
        )
    # Load and split the source documents
    docs = load_documents_from_my_sources(sources_path)
    # Create a new vector store from these documents
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(persistent_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


# Define a state type to be used throughout the pipeline, useful for type checking.
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def build_prompt_template() -> ChatPromptTemplate:
    """
    Builds a chat prompt template for the LLM.

    This function defines a template with placeholders:
    - {user}: The user's name.
    - {question}: The question to be answered.
    - {context}: Retrieved context to support the answer.

    Returns:
        ChatPromptTemplate: The constructed prompt template.
    """
    messages = [
        (
            "human",
            """
            You are {user}'s personal assistant.
            You are in charge of question-answering tasks related to their experience with Software engineering and Computer Science.
            Use the following pieces of retrieved context to answer the question. 
            The context is taken from {user}'s CV, Linkedin and other relevant documents.
            If you don't know the answer, just say that you don't know. Use 4 sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            """,
        )
    ]
    return ChatPromptTemplate(messages)


def testing_llm() -> None:
    """
    Tests the complete RAG pipeline.

    Steps:
    1. Initialize the LLM using the specified model.
    2. Build the prompt template.
    3. Create the vector database for document retrieval.
    4. Define a 'retrieve' function to fetch context from the DB.
    5. Define a 'generate' function to create an answer using retrieved context.
    6. Build a state graph that sequences the 'retrieve' and 'generate' functions.
    7. Invoke the graph with a sample question and print the context and final answer.
    """
    # Initialize the chat model (LLM)
    llm = init_chat_model(model, model_provider=model_provider)
    # Build the prompt that will be sent to the LLM
    prompt = build_prompt_template()
    # Create or load the vector database
    db = create_vector_db()

    def retrieve(state: State) -> dict:
        """
        Retrieve function for the pipeline.

        Given a question from the state, this function uses the vector database
        to find relevant document chunks and returns them in the 'context' field.
        """
        retriever = db.as_retriever(
            search_type=db_search_type,
            search_kwargs=db_search_kwargs,
        )
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State) -> dict:
        """
        Generate function for the pipeline.

        It:
          - Joins the retrieved document chunks into a single string.
          - Invokes the prompt template with the question, context, and user information.
          - Calls the LLM to generate an answer.
          - Returns the generated answer.
        """
        # Combine all retrieved document chunks into one string.
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # Prepare the message based on the prompt template.
        messages = prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
                # User's name hardcoded for this example.
                "user": user,
            }
        )
        # Invoke the language model to generate the answer.
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Build the pipeline (graph) by sequentially chaining the 'retrieve' and 'generate' functions.
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Test the pipeline with a sample question.
    response = graph.invoke({"question": f"Should I hire {user} as an AI Engineer?"})
    print(f"Context: {response['context']}\n\n")
    print(f"Answer: {response['answer']}")


if __name__ == "__main__":
    # Run the pipeline test when this script is executed.
    try:
        testing_llm()
    except Exception as e:
        handle_exception(e, model)
