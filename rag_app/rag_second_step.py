"""
Workshop Guide: Building a Retrieval-Augmented Generation (RAG) Pipeline — Part 2

This file demonstrates more document processing and RAG pipeline construction:
1. Loading and splitting documents from a source directory.
2. Creating a persistable vector database using Chroma.
3. Building a prompt template for a Q&A system.
4. Defining the application state via TypedDicts.
5. Constructing a multi-step pipeline that:
   - Analyzes the input query.
   - Retrieves the most relevant document chunks.
   - Generates an answer by invoking an LLM.
6. Streaming the output for real-time debugging.
"""

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from typing_extensions import List, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# Model configurations for LLM and embedding generation.
model = "llama3.1:8b"
embedding_model = "llama3.1:8b"
model_provider = "ollama"
user = "Antonio Badilla-Olivas"


def load_documents_from_my_sources(sources_path: Path) -> list:
    """
    Loads documents from a specified directory and splits them into chunks.

    Steps:
    1. Iterate over files in the given directory that end with '.pdf' or '.txt'.
    2. Load each document using the appropriate loader.
    3. Attach source metadata (file name) to each document.
    4. Use a text splitter to break documents into smaller chunks.
    5. Annotate each chunk with a 'section' metadata label (beginning, middle, end).

    Args:
        sources_path (Path): Path to the directory containing source documents.

    Returns:
        list: A list of document chunks with associated metadata.
    """
    # Collect all files with .pdf or .txt extensions.
    source_documents = [
        {"file": f, "extension": f.suffix}
        for f in sources_path.iterdir()
        if f.suffix in {".pdf", ".txt"}
    ]

    docs_parsed = []
    # Load and parse each document file.
    for source in source_documents:
        file_path = sources_path / source["file"]
        match source["extension"]:
            case ".txt":
                loader = TextLoader(file_path)
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case _:
                continue  # Skip unsupported file types.
        documents = loader.load()
        # Attach metadata with file name.
        for doc in documents:
            doc.metadata = {"source": source["file"].name}
        docs_parsed.extend(documents)

    # Split the full documents into smaller text chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs_parsed)
    total_documents = len(all_splits)
    third = total_documents // 3

    # Label each chunk with its relative section in the original document.
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
    1. Identify the current directory and set the sources and persistent storage paths.
    2. Create an embeddings instance using OllamaEmbeddings.
    3. If a persistent directory exists, load the existing vector store.
    4. Otherwise, load and split the source documents and create a new vector store.
    
    Returns:
        Chroma: An instance of the vector database.
    """
    current_dir = Path(__file__).resolve().parent
    sources_path = current_dir / "my_sources"
    persistent_directory = current_dir / "db"

    embeddings = OllamaEmbeddings(model=embedding_model)
    # If the vector store already exists, load it.
    if persistent_directory.exists():
        print("Using existing db")
        return Chroma(
            persist_directory=str(persistent_directory),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # Otherwise, create a new vector store.
    print("Persistent directory does not exist. Initializing vector store...")
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


# Define a structure for the search query using a TypedDict.
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]

# Define the overall state structure for the RAG pipeline.
class State(TypedDict):
    query: Search
    question: str
    context: List[Document]
    answer: str


def build_prompt_template() -> ChatPromptTemplate:
    """
    Constructs a chat prompt template for the LLM.

    The prompt instructs the assistant on how to answer design-related questions,
    providing context from the user's documents. Placeholders for user, question,
    and context are specified here.

    Returns:
        ChatPromptTemplate: The prepared prompt template.
    """
    messages = [
        (
            "human",
            """
            You are {user}'s personal assistant.
            You are in charge of question-answering tasks related to their experience with Software engineering and Computer Science.
            Use the following pieces of retrieved context to answer the question. 
            The context is taken from {user}'s CV, Linkedin and other relevant documents.
            If you don't know the answer, just say that you don't know. Sources must sustain your words. Use 4 sentences maximum and keep the answer concise.
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
    3. Create or load the vector database.
    4. Define three key functions:
       - analyze_query: Processes the input question to extract a structured query.
       - retrieve: Retrieves relevant document chunks from the vector store.
       - generate: Forms the final prompt using the retrieved context and queries the LLM for an answer.
    5. Construct a stateful processing graph chaining these functions.
    6. Stream and print each pipeline step for debugging.
    """
    # Initialize the chat language model.
    llm = init_chat_model(model, model_provider=model_provider)
    # Build prompt template to structure the LLM input.
    prompt = build_prompt_template()
    # Create or load the vector database.
    db = create_vector_db()

    def analyze_query(state: State):
        """
        Analyzes the input question to produce a structured search query.

        Uses the language model with structured output to format or enhance the query.
        If no structured query is produced, falls back to the original question.
        """
        structured_llm = llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        if query is None:
            return {"query": state["question"]}
        return {"query": query}

    def retrieve(state: State):
        """
        Retrieves relevant document chunks from the vector database based on the query.

        It uses a similarity-based retriever to find documents that match the search query.
        """
        query = state["query"]
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.01},
        )
        retrieved_docs = retriever.invoke(query)
        return {"context": retrieved_docs}

    def generate(state: State) -> dict:
        """
        Generates an answer using the LLM.

        It:
          1. Joins the retrieved document chunks into a single context string.
          2. Invokes the prompt template with the question, context, and user information.
          3. Calls the LLM to produce the final answer.
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
                "user": user,
            }
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Build the processing graph by chaining analyze_query, retrieve, and generate.
    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    # Stream and print each step of the pipeline for real-time debugging.
    for step in graph.stream(
        {"question": "Should I hire Antonio as a AI Engineer?"},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")


if __name__ == "__main__":
    # Run the pipeline test when the script is executed directly.
    testing_llm()
