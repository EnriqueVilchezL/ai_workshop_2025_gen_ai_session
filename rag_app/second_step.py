from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing import Literal
from typing_extensions import Annotated


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


class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]

class State(TypedDict):
    query: Search
    question: str
    context: List[Document]
    answer: str


def build_prompt_template() -> ChatPromptTemplate:
    messages = [
        (
            "human",
            """
            You are an {user}'s personal assistant.
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
    llm = init_chat_model(model, model_provider=model_provider)
    prompt = build_prompt_template()
    db = create_vector_db()

    def analyze_query(state: State):
        structured_llm = llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        if query is None:
            return {"query": state["question"]}
        return {"query": query}

    def retrieve(state: State):
        query = state["query"]
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.01},
        )
        retrieved_docs = retriever.invoke(query)
        return {"context": retrieved_docs}

    def generate(state: State) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
                "user": "Antonio Badilla-Olivas",
            }
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    for step in graph.stream(
        {"question": "Should I hire Antonio as a AI Engineer?"},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")


if __name__ == "__main__":
    testing_llm()
